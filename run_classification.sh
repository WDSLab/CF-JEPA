#!/bin/bash
# ============================================================
# Run classification experiments (checkpoint mode)
# Checks results/cls_results.csv and only runs what's missing
#
# Usage:
#   bash run_classification.sh --check                                    # Check status
#   bash run_classification.sh --ngpu 2                                   # Run all missing
#   bash run_classification.sh --ucr_only --ngpu 2                        # UCR only
#   bash run_classification.sh --uea_only --ngpu 2                        # UEA only
#   bash run_classification.sh --method ts2vec --ngpu 2                   # Specific method
#   bash run_classification.sh --datasets_file splits/datasets_ucr_t1.txt --ngpu 2 # Subset
# ============================================================

cd "$(dirname "$0")"
python run_classification.py "$@"
