"""
Check and run remaining anomaly detection experiments.

Checks the result CSVs and only runs what's missing. Univariate rows live in
results/ano_results.csv; multivariate rows (one training evaluates both scorings
and every evaluation protocol) live in results/ano_mv_results.csv.

Usage:
    python run_anomaly.py --check                              # Check status only
    python run_anomaly.py                                      # Run all missing (univariate)
    python run_anomaly.py --method ts2vec                      # Specific method
    python run_anomaly.py --method ts2vec --ngpu 2             # Multi-GPU
    python run_anomaly.py --datasets kpi yahoo                 # Specific datasets
    python run_anomaly.py --datasets SMD MSL SMAP PSM          # Multivariate datasets
    python run_anomaly.py --multivariate                       # All multivariate (SMD/MSL/SMAP/PSM)
    python run_anomaly.py --all                                # Univariate + multivariate
    python run_anomaly.py --scoring gaussian                   # Scoring method
    python run_anomaly.py --exclude simmtm                     # Exclude methods
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.launcher import (
    add_common_args, parse_run_plan, print_seed_status, run_jobs, select_methods,
)
from utils.result_status import ANO_CSV, ANO_MV_CSV, ano_done, dims_for

METHODS = [
    'ts2vec', 'softclt', 'trep',
    'infots', 'cost', 'ts_jepa', 'fei', 'autotcl',
    'cf_jepa', 'timesurl',
]
# SimMTM cannot produce per-timestep representations
EXCLUDED_DEFAULT = ['simmtm']

DATASETS_UV = ['kpi', 'yahoo']
DATASETS_MV = ['SMD', 'MSL', 'SMAP', 'PSM']
SCORINGS = ['knn', 'gaussian']
RUNNER = 'scripts/benchmarks/run_anomaly.py'


def _csv_for(dataset, eval_csv=None):
    if eval_csv is not None:
        return eval_csv
    return ANO_MV_CSV if dataset in DATASETS_MV else ANO_CSV


def check_status(methods, datasets, scorings, seeds, max_epoch,
                 hidden_dims=None, output_dims=None, eval_csv=None):
    """Which (method, dataset, scoring, seed) combos are incomplete."""
    missing = []
    for method in methods:
        h, r = dims_for(method, hidden_dims, output_dims)
        for dataset in datasets:
            # Multivariate: one training evaluates both scorings.
            ds_scorings = (['both'] if dataset in DATASETS_MV and scorings == SCORINGS
                           else scorings)
            for scoring in ds_scorings:
                for seed in seeds:
                    if not ano_done(_csv_for(dataset, eval_csv), method, dataset,
                                    seed, max_epoch, scoring, h, r):
                        missing.append({'method': method, 'dataset': dataset,
                                        'scoring': scoring, 'seed': seed})
    return missing


def run_missing(missing, args, checkpoint_epochs):
    """Run missing experiments, one job per (method, dataset, scoring)."""
    groups = {(m['method'], m['dataset'], m['scoring']) for m in missing}
    method_order = {m: i for i, m in enumerate(METHODS)}

    jobs = []
    for method, dataset, scoring in sorted(groups, key=lambda x: (method_order.get(x[0], 999),) + x[1:]):
        cmd = ['--method', method, '--dataset', dataset, '--scoring', scoring,
               '--iterations', str(args.iterations),
               '--checkpoint_epochs', ','.join(str(e) for e in checkpoint_epochs)]
        forwarded = [('--hidden_dims', args.hidden_dims), ('--output_dims', args.output_dims),
                     ('--eval_csv', args.eval_csv)]
        if dataset in DATASETS_MV:
            forwarded += [('--delay', args.delay), ('--window_size', args.window_size),
                          ('--anomaly_ratio', args.anomaly_ratio)]
        for flag, value in forwarded:
            if value is not None:
                cmd += [flag, str(value)]
        jobs.append((f'{method} | {dataset} | scoring: {scoring}', cmd))

    run_jobs(jobs, RUNNER, args.ngpu)


def main():
    parser = argparse.ArgumentParser(description='Check and run remaining anomaly detection experiments')
    add_common_args(parser, checkpoint_default='50,100,200', eval_csv_default=None,
                    eval_csv_help=f'Output CSV path (default: {ANO_CSV} for univariate, '
                                  f'{ANO_MV_CSV} for multivariate).')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets (e.g., kpi yahoo SMD MSL SMAP PSM)')
    parser.add_argument('--multivariate', action='store_true',
                        help='Run all multivariate datasets (SMD/MSL/SMAP/PSM).')
    parser.add_argument('--all', action='store_true',
                        help='Run univariate (kpi/yahoo) + multivariate.')
    parser.add_argument('--window_size', type=int, default=None,
                        help='Sliding window length for multivariate (forwarded to benchmark).')
    parser.add_argument('--anomaly_ratio', type=float, default=None,
                        help='Top-fraction threshold for multivariate (forwarded to benchmark).')
    parser.add_argument('--delay', type=int, default=None,
                        help='Delay tolerance for multivariate (forwarded; benchmark default 7).')
    parser.add_argument('--scoring', type=str, default=None, choices=SCORINGS,
                        help='Specific scoring method only')
    args = parser.parse_args()

    seeds, checkpoint_epochs = parse_run_plan(parser, args)
    methods = select_methods(args, METHODS, always_excluded=EXCLUDED_DEFAULT)
    if args.datasets:
        datasets = args.datasets
    elif args.all:
        datasets = DATASETS_UV + DATASETS_MV
    elif args.multivariate:
        datasets = DATASETS_MV
    else:
        datasets = DATASETS_UV
    scorings = [args.scoring] if args.scoring else SCORINGS

    def check():
        return check_status(methods, datasets, scorings, seeds, max(checkpoint_epochs),
                            args.hidden_dims, args.output_dims, args.eval_csv)

    def show(missing):
        print_seed_status('Anomaly Detection Experiment Status',
                          [f'Checkpoints: {checkpoint_epochs}', f'Seeds: {seeds}',
                           f'Scoring: {SCORINGS}'],
                          missing, keys=('method', 'dataset', 'scoring'),
                          combo_label='method-dataset-scoring')

    missing = check()
    show(missing)
    if args.check:
        return
    if not missing:
        print("\nNothing to run!")
        return
    print(f"\n>>> Running {len(missing)} missing anomaly detection jobs...")
    run_missing(missing, args, checkpoint_epochs)
    print("\n\n>>> Final status:")
    show(check())


if __name__ == '__main__':
    main()
