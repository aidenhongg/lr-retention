import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location("pilot_pod_tool", Path(__file__).parents[1] / "runpod" / "pod.py")
pod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pod)


class FakeClient:
    def __init__(self, *, quote=1.8, returned=1.8, balance=20.0):
        self.quote = quote
        self.returned = returned
        self.balance = balance
        self.calls = []
        self.monitor_error = None

    def query(self, query, variables=None):
        self.calls.append((query, variables))
        if "gpuTypes" in query:
            return {"gpuTypes": [{"id": "GPU", "secureCloud": True, "securePrice": self.quote, "memoryInGb": 80}]}
        if "myself" in query:
            if self.monitor_error:
                raise self.monitor_error
            return {"myself": {"clientBalance": self.balance, "currentSpendPerHr": 1.0,
                               "pods": [{"id": "unrelated", "name": "user-owned", "desiredStatus": "RUNNING", "costPerHr": 1.0}]}}
        if "podFindAndDeployOnDemand" in query:
            return {"podFindAndDeployOnDemand": {"id": "owned-pod", "costPerHr": self.returned}}
        if "podTerminate" in query:
            return {"podTerminate": None}
        if "Status" in query:
            return {"pod": {"id": "owned-pod", "name": "pilot", "desiredStatus": "RUNNING", "costPerHr": self.returned,
                            "runtime": {"uptimeInSeconds": 123, "gpus": [{"id": "gpu", "gpuUtilPercent": 95, "memoryUtilPercent": 80}],
                                        "ports": [{"ip": "203.0.113.1", "privatePort": 22, "publicPort": 12345, "isIpPublic": True}]}}}
        raise AssertionError("Unexpected fake API query")


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    monkeypatch.setattr(pod.time, "time", lambda: 1000.0)
    public_key = tmp_path / "key.pub"
    public_key.write_text("ssh-ed25519 AAAAB3NzaTest example")
    state_path = tmp_path / "pod-state.json"
    client = FakeClient()
    state = pod.create_pod(client, "pilot", "GPU", state_path=state_path, public_key_path=public_key)
    return client, state_path, state


def terminated_ids(client):
    return [variables["input"]["podId"] for query, variables in client.calls if "podTerminate" in query]


def silent(*args, **kwargs):
    pass


def test_create_persists_before_return_and_prices_disk_conservatively(deployment):
    client, state_path, state = deployment
    assert pod.read_state(state_path) == state
    assert state["disk_hourly_rate"] == pytest.approx(1000 * 0.1 / 720)
    assert state["hourly_rate"] == pytest.approx(1.8 + 1000 * 0.1 / 720)
    assert (state["deadline"] - state["created_at"]) / 3600 * state["hourly_rate"] == pytest.approx(14)
    request = next(variables["input"] for query, variables in client.calls if "podFindAndDeployOnDemand" in query)
    assert request["cloudType"] == "SECURE" and request["gpuCount"] == 1
    assert request["containerDiskInGb"] == 1000 and request["volumeInGb"] == 0
    assert request["ports"] == "22/tcp" and request["startSsh"] is True
    assert request["env"] == [{"key": "PUBLIC_KEY", "value": "ssh-ed25519 AAAAB3NzaTest example"}]
    assert "PUBLIC_KEY" not in state_path.read_text()


def test_network_volume_is_attached_priced_and_preserved_on_termination(tmp_path):
    public_key = tmp_path / "key.pub"
    public_key.write_text("ssh-ed25519 AAAA")
    state_path = tmp_path / "state.json"
    client = FakeClient()
    state = pod.create_pod(client, "pilot", "GPU", state_path=state_path, public_key_path=public_key,
                           network_volume_id="durable-volume", network_volume_gb=80)
    request = next(variables["input"] for query, variables in client.calls if "podFindAndDeployOnDemand" in query)
    assert request["networkVolumeId"] == "durable-volume" and request["volumeMountPath"] == "/workspace"
    assert state["network_volume_monthly_usd"] == pytest.approx(5.6)
    assert state["network_volume_hourly_rate"] == pytest.approx(5.6 / 720)
    assert state["hourly_rate"] == pytest.approx(1.8 + 1000 * 0.1 / 720 + 5.6 / 720)
    pod.terminate_pod(client, state_path)
    assert terminated_ids(client) == ["owned-pod"]
    assert not any("networkVolumeDelete" in query for query, _ in client.calls)


@pytest.mark.parametrize("quote,balance", [(2.51, 20), (1.8, 3.0)])
def test_create_refuses_price_or_credit_before_deployment(tmp_path, quote, balance):
    public_key = tmp_path / "key.pub"
    public_key.write_text("ssh-ed25519 AAAA")
    client = FakeClient(quote=quote, balance=balance)
    with pytest.raises(pod.PodError):
        pod.create_pod(client, "pilot", "GPU", state_path=tmp_path / "state.json", public_key_path=public_key)
    assert not any("podFindAndDeployOnDemand" in query for query, _ in client.calls)


def test_unexpected_deployed_price_terminates_only_new_pod(tmp_path):
    public_key = tmp_path / "key.pub"
    public_key.write_text("ssh-ed25519 AAAA")
    client = FakeClient(returned=2.6)
    state_path = tmp_path / "state.json"
    with pytest.raises(pod.PodError, match="exceeded the limit"):
        pod.create_pod(client, "pilot", "GPU", state_path=state_path, public_key_path=public_key)
    assert terminated_ids(client) == ["owned-pod"]
    assert pod.read_state(state_path)["termination_reason"] == "deployed_rate_exceeds_limit"


def test_state_cannot_be_reused_for_another_live_pod(deployment, tmp_path):
    client, state_path, _ = deployment
    initial_calls = len(client.calls)
    with pytest.raises(pod.PodError, match="already owns"):
        pod.create_pod(client, "second", "GPU", state_path=state_path)
    assert len(client.calls) == initial_calls


def test_failed_state_persistence_terminates_the_new_owned_pod(tmp_path, monkeypatch):
    public_key = tmp_path / "key.pub"
    public_key.write_text("ssh-ed25519 AAAA")
    state_path = tmp_path / "state.json"
    original = pod._write_state
    attempts = []
    def fail_once(path, state):
        attempts.append(state["pod_id"])
        if len(attempts) == 1:
            raise OSError("storage unavailable")
        return original(path, state)
    monkeypatch.setattr(pod, "_write_state", fail_once)
    client = FakeClient()
    with pytest.raises(OSError):
        pod.create_pod(client, "pilot", "GPU", state_path=state_path, public_key_path=public_key)
    assert terminated_ids(client) == ["owned-pod"]
    assert pod.read_state(state_path)["terminated_at"] is not None


def test_status_has_ssh_and_gpu_metrics_without_private_fields(deployment):
    client, _, state = deployment
    result = pod.status(client, state)
    assert result["ssh"] == {"host": "203.0.113.1", "port": 12345, "user": "root"}
    assert result["gpus"][0]["gpuUtilPercent"] == 95
    assert result["uptime_seconds"] == 123


def test_budget_guard_stops_before_cap_without_touching_other_pods(deployment):
    client, state_path, state = deployment
    result = pod.guard(client, state_path, now=lambda: state["deadline"] - 20, emit=silent)
    assert result == "budget_limit"
    assert terminated_ids(client) == ["owned-pod"]
    assert pod.read_state(state_path)["terminated_at"] is not None


def test_credit_guard_protects_account_reserve(deployment):
    client, state_path, _ = deployment
    client.balance = 0.9
    assert pod.guard(client, state_path, now=lambda: 1000, emit=silent) == "account_credit_limit"
    assert terminated_ids(client) == ["owned-pod"]


def test_stale_reported_credit_does_not_hide_accrued_account_charges(deployment):
    client, state_path, state = deployment
    state["account_balance_at_creation"] = 4.0
    pod._write_state(state_path, state)
    client.balance = 20.0
    assert pod.guard(client, state_path, now=lambda: 4600, emit=silent) == "account_credit_limit"
    assert terminated_ids(client) == ["owned-pod"]
    assert pod.read_state(state_path)["termination_reason"] == "account_credit_limit"


def test_conservative_credit_keeps_highest_seen_rate_and_ignores_implicit_topups(deployment):
    _, _, state = deployment
    state["max_observed_account_total_rate"] = 6.0
    available, rate = pod._conservative_credit(state, reported_balance=50.0, account_rate=2.0, elapsed=3600)
    assert rate == 6.0
    assert available == pytest.approx(state["account_balance_at_creation"] - 6.0)


def test_guard_logs_heartbeat_then_responds_to_explicit_stop_file(deployment, tmp_path):
    client, state_path, _ = deployment
    stop = tmp_path / "stop-owned-pod"
    completed = tmp_path / "training-complete"
    completed.touch()
    clock = [1000.0]
    def sleep(seconds):
        assert seconds == 30
        clock[0] += seconds
        stop.touch()
    assert pod.guard(client, state_path, stop_file=stop, now=lambda: clock[0], sleep=sleep, emit=silent) == "stop_file"
    events = [json.loads(line) for line in state_path.with_suffix(".telemetry.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "heartbeat"
    assert events[-1]["reason"] == "stop_file"
    assert terminated_ids(client) == ["owned-pod"]


def test_three_monitor_failures_fail_closed_and_sanitize_exception_text(deployment):
    client, state_path, _ = deployment
    client.monitor_error = RuntimeError("secret-runpod-key https://api.runpod.io/graphql?api_key=secret")
    assert pod.guard(client, state_path, now=lambda: 1000, sleep=lambda _: None, emit=silent) == "monitor_failed_three_times"
    telemetry = state_path.with_suffix(".telemetry.jsonl").read_text()
    assert "secret-runpod-key" not in telemetry and "api_key=" not in telemetry
    assert telemetry.count('"event": "monitor_error"') == 3
    assert terminated_ids(client) == ["owned-pod"]


def test_termination_failures_are_bounded_and_remain_explicit(deployment):
    client, state_path, _ = deployment
    client.balance = 0.5
    original = client.query
    attempts = []
    def failing_termination(query, variables=None):
        if "podTerminate" in query:
            attempts.append(variables["input"]["podId"])
            raise RuntimeError("provider error with secret content")
        return original(query, variables)
    client.query = failing_termination
    with pytest.raises(pod.TerminationError, match="three attempts"):
        pod.guard(client, state_path, now=lambda: 1000, sleep=lambda _: None, emit=silent)
    assert attempts == ["owned-pod"] * 3
    assert pod.read_state(state_path).get("terminated_at") is None


def test_closed_stdout_does_not_disable_budget_enforcement(deployment):
    client, state_path, state = deployment
    def broken_output(*args, **kwargs):
        raise BrokenPipeError("closed output")
    assert pod.guard(client, state_path, now=lambda: state["deadline"], emit=broken_output) == "budget_limit"
    assert terminated_ids(client) == ["owned-pod"]


def test_termination_is_idempotent_and_requires_owned_state(deployment):
    client, state_path, _ = deployment
    pod.terminate_pod(client, state_path)
    pod.terminate_pod(client, state_path)
    assert terminated_ids(client) == ["owned-pod"]
    invalid = json.loads(state_path.read_text())
    invalid["owner"] = "someone-else"
    state_path.write_text(json.dumps(invalid))
    with pytest.raises(pod.PodError, match="unowned"):
        pod.terminate_pod(client, state_path)
    assert terminated_ids(client) == ["owned-pod"]


def test_client_uses_auth_header_and_never_exposes_endpoint_error_secrets(monkeypatch):
    requests = []
    def fail(request, timeout):
        requests.append(request)
        raise urllib.error.URLError("https://api.runpod.io/graphql?api_key=super-secret")
    monkeypatch.setattr(pod.urllib.request, "urlopen", fail)
    with pytest.raises(pod.PodError) as captured:
        pod.RunPodClient("super-secret").query("query { myself { clientBalance } }")
    assert requests[0].full_url == "https://api.runpod.io/graphql"
    assert requests[0].headers["Authorization"] == "Bearer super-secret"
    assert "super-secret" not in str(captured.value)


def test_graphql_error_details_are_not_printed(monkeypatch):
    class Response(io.StringIO):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
    monkeypatch.setattr(pod.urllib.request, "urlopen", lambda *args, **kwargs: Response(
        '{"errors":[{"message":"secret-key embedded in provider error"}]}'))
    with pytest.raises(pod.PodError) as captured:
        pod.RunPodClient("private").query("query { anything }")
    assert "secret-key" not in str(captured.value)
