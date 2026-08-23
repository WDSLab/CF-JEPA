#!/bin/bash
# ============================================================
# Run anomaly detection experiments (checkpoint mode)
# Checks ano_results.csv and only runs what's missing
#
# Usage:
#   bash run_anomaly.sh --check                              # Check status
#   bash run_anomaly.sh                                      # Run all missing
#   bash run_anomaly.sh --method ts2vec --ngpu 2             # Specific method
#   bash run_anomaly.sh --datasets kpi yahoo                 # Specific datasets
#   bash run_anomaly.sh --scoring knn                        # Specific scoring
#   bash run_anomaly.sh --exclude simmtm,ts_jepa             # Exclude methods
# ============================================================

cd "$(dirname "$0")"
python run_anomaly.py "$@"
