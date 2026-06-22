#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${SCRIPT_DIR}/clean_reference_detector.py" \
  --run-dir "${1:-${PROJECT_ROOT}/saved_models/model_CifarFedProtopnet_Jun.03_19.36.29_cifar10_mprobe_non_iid_epoch10_data_washing_seed0}" \
  --audit-dir "${2:-${SCRIPT_DIR}/output/cifar10_non_iid_mprobe_epoch10_audit}" \
  --output-dir "${3:-${SCRIPT_DIR}/output/clean_reference_detector}" \
  --seed 0 \
  --epochs 5 \
  --batch-size 256 \
  --score-mode all_suspicious_poison \
  --threshold-side all \
  --threshold-quantile 0.50
