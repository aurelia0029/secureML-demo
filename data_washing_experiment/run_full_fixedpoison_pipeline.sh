#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SAVED_DIR="${PROJECT_ROOT}/saved_models"
OUTPUT_DIR="${SCRIPT_DIR}/output"
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
PIPELINE_RUN_DIR="${OUTPUT_DIR}/full_fixedpoison_pipeline_${TIMESTAMP}"
PIPELINE_LOG="${PIPELINE_RUN_DIR}/log.txt"
TMP_LOG="${PIPELINE_RUN_DIR}/tmp.log"
AUDIT_DIR="${SCRIPT_DIR}/output/cifar10_non_iid_mprobe_epoch10_fixedpoison_audit"
DETECTOR_OUTPUT_DIR="${PIPELINE_RUN_DIR}/materialized_trigger_detector"
RUN_NAME_KEYWORD="cifar10_mprobe_non_iid_epoch10_fixedpoison_data_washing_seed0"
mkdir -p "${PIPELINE_RUN_DIR}"

PIPELINE_START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S %Z')"
PIPELINE_START_EPOCH="$(date +%s)"

run_and_log() {
  "$@" 2>&1 | tee -a "${TMP_LOG}"
}

find_latest_run_dir() {
  find "${SAVED_DIR}" -maxdepth 1 -type d -name "*${RUN_NAME_KEYWORD}" | sort | tail -n 1
}

{
  echo "[pipeline] full_fixedpoison_pipeline"
  echo "[time] pipeline_start=${PIPELINE_START_HUMAN}"
  echo "[path] pipeline_run_dir=${PIPELINE_RUN_DIR}"
  echo "[path] audit_dir=${AUDIT_DIR}"
  echo "[path] detector_output_dir=${DETECTOR_OUTPUT_DIR}"
} | tee "${TMP_LOG}"

BEFORE_RUN_DIR="$(find_latest_run_dir || true)"
{
  echo "[path] previous_run_dir=${BEFORE_RUN_DIR:-<none>}"
  echo "[phase] training_global_model start"
} | tee -a "${TMP_LOG}"

TRAIN_START_EPOCH="$(date +%s)"
run_and_log python "${PROJECT_ROOT}/training.py" \
  --name "${RUN_NAME_KEYWORD}" \
  --params "${SCRIPT_DIR}/cifar10_non_iid_mprobe_epoch10_fixedpoison_audit.yaml" \
  --commit "none" \
  --seed 0
TRAIN_END_EPOCH="$(date +%s)"

RUN_DIR="$(find_latest_run_dir)"
if [[ -z "${RUN_DIR}" ]]; then
  echo "[error] could not locate new run directory under ${SAVED_DIR}" | tee -a "${TMP_LOG}"
  exit 1
fi
RUN_LOG_FILE="${RUN_DIR}/log.txt"

# Use the training framework's native log as the main experiment log so the
# pipeline folder keeps the same format and metrics as regular training runs.
cp "${RUN_LOG_FILE}" "${PIPELINE_LOG}"

{
  echo "[phase] training_global_model end"
  echo "[time] training_elapsed_seconds=$((TRAIN_END_EPOCH - TRAIN_START_EPOCH))"
  echo "[path] run_dir=${RUN_DIR}"
  echo "[phase] materialized_trigger_detector start"
} | tee -a "${PIPELINE_LOG}"

DETECT_START_EPOCH="$(date +%s)"
run_and_log python "${SCRIPT_DIR}/materialized_trigger_detector.py" \
  --run-dir "${RUN_DIR}" \
  --audit-dir "${AUDIT_DIR}" \
  --output-dir "${DETECTOR_OUTPUT_DIR}" \
  --threshold-quantile 0.01
DETECT_END_EPOCH="$(date +%s)"

PIPELINE_END_HUMAN="$(date '+%Y-%m-%d %H:%M:%S %Z')"
PIPELINE_END_EPOCH="$(date +%s)"

{
  echo "[phase] materialized_trigger_detector end"
  echo "[time] detector_elapsed_seconds=$((DETECT_END_EPOCH - DETECT_START_EPOCH))"
  echo "[time] pipeline_end=${PIPELINE_END_HUMAN}"
  echo "[time] pipeline_elapsed_seconds=$((PIPELINE_END_EPOCH - PIPELINE_START_EPOCH))"
  echo "[path] detector_summary=${DETECTOR_OUTPUT_DIR}/summary.json"
} | tee -a "${PIPELINE_LOG}"

rm -f "${TMP_LOG}"

echo "${PIPELINE_LOG}"
