#!/bin/bash
# ============================================================
# Run forecast experiments (checkpoint mode)
# Checks fcst_results.csv and only runs what's missing
#
# Usage:
#   bash run_forecast.sh --check                              # Check status
#   bash run_forecast.sh                                      # Run all missing
#   bash run_forecast.sh --method ts2vec --ngpu 2             # Specific method
#   bash run_forecast.sh --univar_only                        # Univariate only
#   bash run_forecast.sh --datasets ETTh1 ETTh2               # Specific datasets
# ============================================================

cd "$(dirname "$0")"
python run_forecast.py "$@"
