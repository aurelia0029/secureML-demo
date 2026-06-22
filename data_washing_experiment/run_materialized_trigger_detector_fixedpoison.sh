#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="${1:-${PROJECT_ROOT}/saved_models/model_CifarFedProtopnet_Jun.03_20.38.15_cifar10_mprobe_non_iid_epoch10_fixedpoison_data_washing_seed0}"
AUDIT_DIR="${2:-${SCRIPT_DIR}/output/cifar10_non_iid_mprobe_epoch10_fixedpoison_audit}"
OUTPUT_DIR="${3:-${SCRIPT_DIR}/output/materialized_trigger_detector_fixedpoison_${TIMESTAMP}}"
LOG_FILE="${LOG_DIR}/materialized_trigger_detector_fixedpoison_${TIMESTAMP}.log"

START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S %Z')"
START_EPOCH="$(date +%s)"

{
  echo "[run] materialized_trigger_detector_fixedpoison"
  echo "[time] start=${START_HUMAN}"
  echo "[path] run_dir=${RUN_DIR}"
  echo "[path] audit_dir=${AUDIT_DIR}"
  echo "[path] output_dir=${OUTPUT_DIR}"

  python -u "${SCRIPT_DIR}/materialized_trigger_detector.py" \
    --run-dir "${RUN_DIR}" \
    --audit-dir "${AUDIT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --data-path "${PROJECT_ROOT}/.data" \
    --threshold-quantile 0.01
} 2>&1 | tee "${LOG_FILE}"

END_HUMAN="$(date '+%Y-%m-%d %H:%M:%S %Z')"
END_EPOCH="$(date +%s)"
ELAPSED_SECONDS="$((END_EPOCH - START_EPOCH))"

{
  echo "[time] end=${END_HUMAN}"
  echo "[time] elapsed_seconds=${ELAPSED_SECONDS}"
} | tee -a "${LOG_FILE}"

echo "${LOG_FILE}"
