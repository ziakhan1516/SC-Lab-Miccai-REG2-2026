#!/usr/bin/env bash
# Load evaluation_config.env and derived host paths. Source from do_*.sh scripts.

_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_CONFIG_ENV="${_CONFIG_DIR}/evaluation_config.env"

if [[ ! -f "${_CONFIG_ENV}" ]]; then
  echo "Missing ${_CONFIG_ENV}" >&2
  exit 1
fi

# shellcheck source=evaluation_config.env
set -a
# shellcheck disable=SC1090
source "${_CONFIG_ENV}"
set +a

SCRIPT_DIR="${_CONFIG_DIR}"
GROUND_TRUTH_HOST="${SCRIPT_DIR}/ground_truth"
JUDGE_MODEL_HOST="${GROUND_TRUTH_HOST}/${JUDGE_MODEL_REL}"
