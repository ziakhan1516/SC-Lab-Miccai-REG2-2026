#!/usr/bin/env bash
# Invoke from package root: ./prepare_test_ground_truth.sh
set -euo pipefail
_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DATA_DIR="${_ROOT_DIR}/data"
cd "${_DATA_DIR}"
exec python3 "${_DATA_DIR}/prepare_test_ground_truth.py" "$@"
