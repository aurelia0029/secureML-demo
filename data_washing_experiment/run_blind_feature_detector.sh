#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_DIR="${1:-${PROJECT_ROOT}/saved_models/model_CifarFedProtopnet_Jun.04_02.55.00_cifar10_mprobe_non_iid_epoch10_fixedpoison_data_washing_seed0}"
AUDIT_DIR="${2:-${SCRIPT_DIR}/output/cifar10_non_iid_mprobe_epoch10_fixedpoison_audit}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT_DIR="${3:-${SCRIPT_DIR}/output/blind_feature_detector_${TIMESTAMP}}"

cd "${PROJECT_ROOT}"

python "${SCRIPT_DIR}/blind_feature_detector.py" \
  --run-dir "${RUN_DIR}" \
  --audit-dir "${AUDIT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 5 \
  --batch-size 256 \
  --threshold-quantile 0.99 \
  --score-mode pred_dist_plus_uncertainty
