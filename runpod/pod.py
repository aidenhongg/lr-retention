"""RunPod API orchestration and a foreground budget guard; never runs models."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path


IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
OWNER = "yeh-research-budget-pilot"
DISK_GB_MONTH_USD = 0.10
API_TIMEOUT_SECONDS = 20


class PodError(RuntimeError):
    """An error containing only deliberately sanitized diagnostic text."""


class TerminationError(PodError):
    pass


def load_api_key() -> str:
    key = os.environ.get("RUNPOD_KEY", "").strip()
    if not key:
        path = Path(__file__).resolve().parents[2] / ".env"
        if path.is_file():
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                name, separator, value = line.partition("=")
                if separator and name.strip() == "RUNPOD_KEY":
                    key = value.strip().strip("\"'")
                    break
    if not key:
        raise PodError("RUNPOD_KEY is unavailable in the environment or project .env")
    return key


class RunPodClient:
    def __init__(self, key: str | None = None):
        self._key = key or load_api_key()

    def query(self, query: str, variables: dict | None = None) -> dict:
        request = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data=json.dumps({"query": query, "variables": variables or {}}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}",
                     "User-Agent": "yeh-research-budget-guard/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            raise PodError(f"RunPod HTTP request failed with status {error.code}") from None
        except Exception:
            raise PodError("RunPod network request or response decoding failed") from None
        if not isinstance(payload, dict) or payload.get("errors"):
            raise PodError("RunPod rejected the GraphQL request; response details withheld")
        if not isinstance(payload.get("data"), dict):
            raise PodError("RunPod returned an invalid response")
        return payload["data"]


def _number(value, name: str, *, allow_zero: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise PodError(f"Invalid numeric field: {name}")
    return float(value)


def _utc(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat()


def _write_state(path: Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_state(path: Path) -> dict:
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("owner") != OWNER
            or not isinstance(state.get("pod_id"), str) or not re.fullmatch(r"[A-Za-z0-9_-]+", state["pod_id"])):
        raise PodError("Invalid or unowned pod state")
    for field in ("created_at", "hourly_rate", "budget_usd", "reserve_usd", "deadline"):
        _number(state.get(field), field, allow_zero=field == "reserve_usd")
    if state["budget_usd"] <= state["reserve_usd"]:
        raise PodError("Invalid saved spending limit")
    return state


def list_pods(client: RunPodClient) -> dict:
    return client.query("query { myself { clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr } } }")["myself"]


def _terminate_id(client: RunPodClient, pod_id: str) -> None:
    client.query("mutation Terminate($input:PodTerminateInput!) { podTerminate(input:$input) }",
                 {"input": {"podId": pod_id}})


def terminate_pod(client: RunPodClient, state_path: Path, *, reason: str = "manual") -> dict:
    state = read_state(state_path)
    if state.get("terminated_at") is not None:
        return state
    _terminate_id(client, state["pod_id"])
    state.update(terminated_at=time.time(), termination_reason=reason)
    _write_state(state_path, state)
    return state


def create_pod(client: RunPodClient, name: str, gpu: str, *, state_path: Path,
               disk_gb: int = 1000, max_gpu_price: float = 2.5, budget: float = 15.0,
               reserve: float = 1.0, public_key_path: Path | None = None,
               network_volume_id: str | None = None, network_volume_gb: int | None = None) -> dict:
    """Preflight rates/credit, deploy once, and durably save ownership immediately."""
    if Path(state_path).exists() and read_state(state_path).get("terminated_at") is None:
        raise PodError("State already owns an active pod; terminate it before creating another")
    budget = _number(budget, "budget")
    reserve = _number(reserve, "reserve", allow_zero=True)
    maximum = _number(max_gpu_price, "max_gpu_price")
    if budget > 15 or reserve < 1 or budget <= reserve or type(disk_gb) is not int or disk_gb <= 0:
        raise PodError("Pilot budget must be at most 15 USD, reserve at least 1 USD, and disk size positive")
    if not name.strip() or not gpu.strip():
        raise PodError("Pod name and GPU type are required")
    if network_volume_id is not None:
        if (not re.fullmatch(r"[A-Za-z0-9_-]+", network_volume_id)
                or type(network_volume_gb) is not int or not 10 <= network_volume_gb <= 4096):
            raise PodError("Network volume requires its ID and allocated size in GB")
    elif network_volume_gb is not None:
        raise PodError("Network volume size requires a network volume ID")
    public_key = (public_key_path or Path.home() / ".ssh" / "id_ed25519.pub").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"ssh-(?:ed25519|rsa) [A-Za-z0-9+/=]+(?: [^\r\n]+)?", public_key):
        raise PodError("Expected an existing SSH public key")
    gpu_rows = client.query("query GPU($input:GpuTypeFilter) { gpuTypes(input:$input) { id secureCloud securePrice memoryInGb } }",
                            {"input": {"id": gpu}})["gpuTypes"]
    matching = [row for row in gpu_rows if row.get("id") == gpu and row.get("secureCloud")]
    if len(matching) != 1:
        raise PodError("Requested GPU is unavailable in Secure Cloud")
    quoted = _number(matching[0].get("securePrice"), "secure GPU price")
    if quoted > maximum:
        raise PodError("GPU quote exceeds the configured hourly price limit")
    disk_rate = disk_gb * DISK_GB_MONTH_USD / 720
    volume_monthly = (min(network_volume_gb, 1000) * 0.07 + max(network_volume_gb - 1000, 0) * 0.05) if network_volume_gb else 0
    volume_rate = volume_monthly / 720
    conservative_rate = quoted + disk_rate + volume_rate
    account = list_pods(client)
    balance = _number(account.get("clientBalance"), "account balance", allow_zero=True)
    account_rate = _number(account.get("currentSpendPerHr"), "account spending rate", allow_zero=True)
    if balance <= reserve + conservative_rate + account_rate:
        raise PodError("Insufficient account credit for one hour of existing and new resources plus reserve")
    started = time.time()
    deployment_input = {"cloudType": "SECURE", "gpuCount": 1, "volumeInGb": 0,
                        "containerDiskInGb": disk_gb, "minVcpuCount": 4, "minMemoryInGb": 40,
                        "gpuTypeId": gpu, "name": name, "imageName": IMAGE, "ports": "22/tcp",
                        "startSsh": True, "supportPublicIp": True,
                        "env": [{"key": "PUBLIC_KEY", "value": public_key}]}
    if network_volume_id is not None:
        deployment_input.update(networkVolumeId=network_volume_id, volumeMountPath="/workspace")
    deployment = client.query(
        "mutation Deploy($input:PodFindAndDeployOnDemandInput!) { podFindAndDeployOnDemand(input:$input) { id costPerHr } }",
        {"input": deployment_input},
    )["podFindAndDeployOnDemand"]
    pod_id = deployment.get("id")
    if not isinstance(pod_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
        raise PodError("RunPod deployment did not return a usable pod ID; inspect the account")
    state = {"schema_version": 1, "owner": OWNER, "pod_id": pod_id, "name": name, "gpu_type": gpu,
             "created_at": started, "created_utc": _utc(started), "disk_gb": disk_gb,
             "disk_hourly_rate": disk_rate, "quoted_gpu_hourly_rate": quoted,
             "network_volume_id": network_volume_id, "network_volume_gb": network_volume_gb,
             "network_volume_hourly_rate": volume_rate, "network_volume_monthly_usd": volume_monthly,
             "max_gpu_price": maximum, "budget_usd": budget, "reserve_usd": reserve,
             "account_balance_at_creation": balance, "account_rate_before_creation": account_rate,
             "hourly_rate": conservative_rate, "deadline": started + (budget - reserve) / conservative_rate * 3600}
    try:
        returned = _number(deployment.get("costPerHr"), "deployed hourly price")
        state["returned_hourly_rate"] = returned
        state["hourly_rate"] = max(quoted, returned) + disk_rate + volume_rate
        state["deadline"] = started + (budget - reserve) / state["hourly_rate"] * 3600
        _write_state(state_path, state)
        if returned > maximum:
            state = terminate_pod(client, state_path, reason="deployed_rate_exceeds_limit")
            raise PodError("Deployed hourly price exceeded the limit; owned pod terminated")
    except Exception:
        if not state.get("terminated_at"):
            try:
                _terminate_id(client, pod_id)
                state.update(terminated_at=time.time(), termination_reason="creation_validation_or_persistence_failed")
                _write_state(state_path, state)
            except Exception:
                raise PodError("Creation safeguard failed; inspect and terminate the newly created pod") from None
        raise
    return state


def status(client: RunPodClient, state: dict) -> dict:
    pod = client.query(
        "query Status($input:PodFilter!) { pod(input:$input) { id name desiredStatus costPerHr "
        "runtime { uptimeInSeconds ports { ip privatePort publicPort isIpPublic } "
        "gpus { id gpuUtilPercent memoryUtilPercent } } } }", {"input": {"podId": state["pod_id"]}},
    )["pod"]
    if pod is None:
        return {"pod_id": state["pod_id"], "desiredStatus": "NOT_FOUND", "ssh": None}
    runtime = pod.get("runtime") or {}
    ssh = next(({"host": port["ip"], "port": port["publicPort"], "user": "root"}
                for port in runtime.get("ports") or [] if port.get("privatePort") == 22 and port.get("isIpPublic")), None)
    return {"pod_id": pod["id"], "name": pod.get("name"), "desiredStatus": pod.get("desiredStatus"),
            "costPerHr": pod.get("costPerHr"), "uptime_seconds": runtime.get("uptimeInSeconds"),
            "gpus": runtime.get("gpus") or [], "ssh": ssh,
            "estimated_cost_usd": max(0, time.time() - state["created_at"]) / 3600 * state["hourly_rate"],
            "conservative_hourly_rate": state["hourly_rate"], "deadline_utc": _utc(state["deadline"])}


def _error_event(error: Exception) -> dict:
    return {"error_type": type(error).__name__, "error": str(error) if isinstance(error, PodError) else "Operation failed; exception text withheld",
            "traceback": [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                          for frame in traceback.extract_tb(error.__traceback__)]}


def _conservative_credit(state: dict, reported_balance: float, account_rate: float, elapsed: float):
    initial = _number(state.get("account_balance_at_creation"), "initial account balance", allow_zero=True)
    previous = _number(state.get("account_rate_before_creation"), "initial account rate", allow_zero=True)
    saved_maximum = _number(state.get("max_observed_account_total_rate", 0), "saved account rate", allow_zero=True)
    maximum = max(previous + state["hourly_rate"], account_rate, saved_maximum)
    estimated = initial - max(0, elapsed) / 3600 * maximum
    return min(reported_balance, estimated), maximum


def guard(client: RunPodClient, state_path: Path, *, poll: float = 30,
          stop_file: Path | None = None, sleep=time.sleep, now=time.time, emit=print) -> str:
    """Watch credit and this pod's spending until manual or budget termination."""
    poll = _number(poll, "poll")
    if poll > 30:
        raise PodError("Guard polling must be at most 30 seconds")
    telemetry_path = Path(state_path).with_suffix(".telemetry.jsonl")
    failures = 0

    def log(event):
        event = {"timestamp": _utc(now()), **event}
        line = json.dumps(event, sort_keys=True, allow_nan=False)
        try:
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with telemetry_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
        except OSError:
            pass
        try:
            emit(line, flush=True)
        except OSError:
            pass

    def finish(reason):
        for attempt in range(3):
            try:
                terminate_pod(client, state_path, reason=reason)
                log({"event": "terminated", "reason": reason, "pod_id": state["pod_id"]})
                return reason
            except Exception as error:
                log({"event": "termination_error", "attempt": attempt + 1, **_error_event(error)})
                if attempt < 2:
                    sleep(min(poll, 5))
        raise TerminationError("Owned pod termination failed after three attempts; immediate account action required")

    while True:
        state = read_state(state_path)
        if state.get("terminated_at") is not None:
            return "already_terminated"
        elapsed = max(0, now() - state["created_at"])
        estimated = elapsed / 3600 * state["hourly_rate"]
        lead_seconds = poll + 3 * API_TIMEOUT_SECONDS
        if (now() + lead_seconds >= state["deadline"]
                or estimated + lead_seconds / 3600 * state["hourly_rate"] >= state["budget_usd"] - state["reserve_usd"]):
            return finish("budget_limit")
        if stop_file is not None and Path(stop_file).exists():
            return finish("stop_file")
        estimated_available, saved_account_rate = _conservative_credit(
            state, state["account_balance_at_creation"], 0, elapsed)
        if estimated_available <= state["reserve_usd"] + saved_account_rate * lead_seconds / 3600:
            return finish("account_credit_limit")
        try:
            account = list_pods(client)
            balance = _number(account.get("clientBalance"), "account balance", allow_zero=True)
            account_rate = _number(account.get("currentSpendPerHr"), "account spending rate", allow_zero=True)
            available, maximum_account_rate = _conservative_credit(state, balance, account_rate, elapsed)
            if maximum_account_rate > state.get("max_observed_account_total_rate", 0):
                state["max_observed_account_total_rate"] = maximum_account_rate
                _write_state(state_path, state)
            if available <= state["reserve_usd"] + maximum_account_rate * lead_seconds / 3600:
                return finish("account_credit_limit")
            current = status(client, state)
            if current["desiredStatus"] in ("EXITED", "TERMINATED", "NOT_FOUND"):
                return finish("pod_no_longer_running")
            observed = _number(current.get("costPerHr"), "observed pod rate")
            if observed > state["max_gpu_price"]:
                return finish("hourly_rate_limit")
            observed_total = observed + state["disk_hourly_rate"] + state.get("network_volume_hourly_rate", 0)
            if observed_total > state["hourly_rate"]:
                state["hourly_rate"] = observed_total
                state["deadline"] = min(state["deadline"], state["created_at"] +
                                         (state["budget_usd"] - state["reserve_usd"]) / state["hourly_rate"] * 3600)
                _write_state(state_path, state)
            failures = 0
            log({"event": "heartbeat", "account_balance_usd": balance,
                 "account_spend_per_hour": account_rate, "conservative_available_credit_usd": available,
                 "max_observed_account_total_rate": maximum_account_rate, **current})
        except TerminationError:
            raise
        except Exception as error:
            failures += 1
            log({"event": "monitor_error", "consecutive_failures": failures, **_error_event(error)})
            if failures >= 3:
                return finish("monitor_failed_three_times")
        sleep(poll)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("name")
    create.add_argument("gpu")
    create.add_argument("--disk-gb", type=int, default=1000)
    create.add_argument("--max-gpu-price", type=float, default=2.5)
    create.add_argument("--budget", type=float, default=15)
    create.add_argument("--reserve", type=float, default=1)
    create.add_argument("--public-key", type=Path)
    create.add_argument("--network-volume")
    create.add_argument("--network-volume-gb", type=int)
    create.add_argument("--state", type=Path, required=True)
    commands.add_parser("list")
    for name in ("status", "terminate", "guard"):
        command = commands.add_parser(name)
        command.add_argument("--state", type=Path, required=True)
        if name == "guard":
            command.add_argument("--poll", type=float, default=30)
            command.add_argument("--stop-file", type=Path)
    args = parser.parse_args(argv)
    try:
        client = RunPodClient()
        if args.command == "create":
            result = create_pod(client, args.name, args.gpu, state_path=args.state, disk_gb=args.disk_gb,
                                max_gpu_price=args.max_gpu_price, budget=args.budget, reserve=args.reserve,
                                public_key_path=args.public_key, network_volume_id=args.network_volume,
                                network_volume_gb=args.network_volume_gb)
        elif args.command == "list":
            result = list_pods(client)
        elif args.command == "status":
            result = status(client, read_state(args.state))
        elif args.command == "terminate":
            result = terminate_pod(client, args.state)
        else:
            result = {"guard_result": guard(client, args.state, poll=args.poll, stop_file=args.stop_file)}
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"event": "error", **_error_event(error)}, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
