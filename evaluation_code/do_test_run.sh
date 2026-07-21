#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

GPU_ARGS=()
if [ "${USE_GPUS}" = "1" ] || { [ "${USE_GPUS}" = "auto" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; }; then
  GPU_ARGS+=(--gpus all)
  USE_GPUS=1
else
  USE_GPUS=0
fi

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

# Inject all runtime settings from evaluation_config.env
DOCKER_ENV_ARGS=(
  -e "GRAND_CHALLENGE_MAX_WORKERS=${GRAND_CHALLENGE_MAX_WORKERS}"
  -e "JUDGE_DEVICE=${JUDGE_DEVICE}"
  -e "JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH}"
  -e "EMBEDDING_MODEL=${EMBEDDING_MODEL}"
  -e "INCLUDE_PER_CASE_RESULTS=${INCLUDE_PER_CASE_RESULTS}"
  -e "INCLUDE_EVALUATION_DETAILS=${INCLUDE_EVALUATION_DETAILS}"
)

cleanup() {
    echo "=+= Cleaning permissions ..."
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      "$DOCKER_IMAGE_TAG" \
      -c "chmod -R -f o+rwX /output/* || true"

    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

chmod -R -f o+rX "$INPUT_DIR" "${GROUND_TRUTH_HOST}"

if [ -d "$OUTPUT_DIR" ]; then
  chmod -f o+rwX "$OUTPUT_DIR"

  echo "=+= Cleaning up any earlier output"
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      "$DOCKER_IMAGE_TAG" \
      -c "rm -rf /output/* || true"
else
  mkdir -m o+rwX "$OUTPUT_DIR"
fi

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

echo "=+= Doing a forward pass (USE_GPUS=${USE_GPUS}, JUDGE_DEVICE=${JUDGE_DEVICE}, JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH}, INCLUDE_PER_CASE_RESULTS=${INCLUDE_PER_CASE_RESULTS}, INCLUDE_EVALUATION_DETAILS=${INCLUDE_EVALUATION_DETAILS})"
docker run --rm "${GPU_ARGS[@]}" \
    --platform=linux/amd64 \
    --network none \
    "${DOCKER_ENV_ARGS[@]}" \
    --volume "$INPUT_DIR":/input:ro \
    --volume "$OUTPUT_DIR":/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    --volume "${GROUND_TRUTH_HOST}":/opt/ml/input/data/ground_truth:ro \
    "$DOCKER_IMAGE_TAG"

echo "=+= Wrote results to ${OUTPUT_DIR}"
