#!/usr/bin/env bash
set -euo pipefail
cd /opt/mock-experiment
python -m venv /opt/pilot-env
source /opt/pilot-env/bin/activate
python -m pip install --disable-pip-version-check -r requirements.txt
python -m pip check
python - <<'PY'
import json
import torch
import transformers
import bitsandbytes
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.is_bf16_supported(), "BF16 unsupported"
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(), "transformers": transformers.__version__,
                  "bitsandbytes": bitsandbytes.__version__}), flush=True)
PY
python -m pytest tests -q
