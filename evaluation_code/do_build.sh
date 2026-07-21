#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG" \
  --build-arg "GRAND_CHALLENGE_MAX_WORKERS=${GRAND_CHALLENGE_MAX_WORKERS}" \
  --build-arg "JUDGE_DEVICE=${JUDGE_DEVICE}" \
  --build-arg "JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH}" \
  --build-arg "EMBEDDING_MODEL=${EMBEDDING_MODEL}" \
  --build-arg "INCLUDE_PER_CASE_RESULTS=${INCLUDE_PER_CASE_RESULTS}" \
  --build-arg "INCLUDE_EVALUATION_DETAILS=${INCLUDE_EVALUATION_DETAILS}" \
  ${DOCKER_QUIET_BUILD:+--quiet} \
  "$SCRIPT_DIR" 2>&1
