#!/usr/bin/env bash
# Download judge LLM weights into ground_truth/<JUDGE_MODEL_REL>/.
#
# Flow: source evaluation_config.env -> temp venv -> pip install huggingface_hub
#       -> snapshot_download(JUDGE_MODEL_REPO_ID, JUDGE_MODEL_HOST) -> remove venv
#
# See README.md for usage.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

if [[ -z "${JUDGE_MODEL_REPO_ID:-}" ]]; then
  echo "JUDGE_MODEL_REPO_ID is not set in evaluation_config.env" >&2
  exit 1
fi

VENV_DIR="$(mktemp -d "${TMPDIR:-/tmp}/reg26-download-venv.XXXXXX")"
cleanup() { rm -rf "${VENV_DIR}"; }
trap cleanup EXIT

echo "Creating temporary Python environment..."
python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install -q "huggingface_hub>=0.26.0"

echo "Downloading ${JUDGE_MODEL_REPO_ID} -> ${JUDGE_MODEL_HOST}"
export JUDGE_MODEL_REPO_ID JUDGE_MODEL_HOST
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

python <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["JUDGE_MODEL_REPO_ID"]
local_dir = os.environ["JUDGE_MODEL_HOST"]
token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None

os.makedirs(local_dir, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    token=token,
)
print(f"Done: {local_dir}")
PY
