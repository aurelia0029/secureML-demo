#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PROJECT_ROOT}/training.py" \
  --name "cifar10_mprobe_non_iid_epoch10_fixedpoison_data_washing_seed0" \
  --params "${SCRIPT_DIR}/cifar10_non_iid_mprobe_epoch10_fixedpoison_audit.yaml" \
  --commit "none" \
  --seed 0
