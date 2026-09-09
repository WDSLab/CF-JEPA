"""
Check and run remaining anomaly detection experiments.

Checks ano_results.csv and only runs what's missing.

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
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import get_method_dims

METHODS = [
    'ts2vec', 'softclt', 'trep',
    'infots', 'cost', 'ts_jepa', 'fei', 'autotcl',
    'cf_jepa', 'timesurl',
]

# SimMTM excluded: cannot produce per-timestep representations
EXCLUDED_DEFAULT = ['simmtm']

DATASETS_UV = ['kpi', 'yahoo']
DATASETS_MV = ['SMD', 'MSL', 'SMAP', 'PSM']
DATASETS = DATASETS_UV  # default = univariate (back-compat)
SCORING_METHODS = ['knn', 'gaussian']
CHECKPOINT_EPOCHS = [50, 100, 200]
MAX_EPOCH = 200
ITERATIONS = 3
SEEDS = [42, 43, 44]


def _is_done(method, dataset, seed, max_epoch, scoring,
             hidden_dims=None, repr_dims=None,
             eval_csv='results/ano_results.csv'):
    """Check if last checkpoint results exist in anomaly CSV."""
    if not os.path.exists(eval_csv):
        return False
    with open(eval_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get('method') == method and
                row.get('dataset') == dataset and
                row.get('seed') == str(seed) and
                row.get('epoch') == str(max_epoch) and
                row.get('hidden_dims', str(hidden_dims)) == str(hidden_dims) and
                row.get('repr_dims', str(repr_dims)) == str(repr_dims) and
                row.get('scoring') == scoring):
                return True
    return False


def check_status(methods=None, datasets=None, scoring_methods=None,
                 hidden_dims=None, repr_dims=None,
                 eval_csv='results/ano_results.csv'):
    """Check which (method, dataset, scoring, seed) combos are incomplete."""
    methods = methods or METHODS
    datasets = datasets or DATASETS
    scoring_methods = scoring_methods or SCORING_METHODS

    missing = []
    for method in methods:
        default_h, default_r = get_method_dims(method)
        actual_h = hidden_dims if hidden_dims is not None else default_h
        actual_r = repr_dims if repr_dims is not None else default_r
        for dataset in datasets:
            for scoring in scoring_methods:
                for seed in SEEDS:
                    if not _is_done(method, dataset, seed, MAX_EPOCH, scoring,
                                    hidden_dims=actual_h, repr_dims=actual_r,
                                    eval_csv=eval_csv):
                        missing.append({
                            'method': method,
                            'dataset': dataset,
                            'scoring': scoring,
                            'seed': seed,
                        })
    return missing


def print_status(missing):
    print("\n" + "=" * 60)
    print(" Anomaly Detection Experiment Status")
    print(f" Checkpoints: {CHECKPOINT_EPOCHS}")
    print(f" Seeds: {SEEDS}")
    print(f" Scoring: {SCORING_METHODS}")
    print("=" * 60)

    if not missing:
        print("\n  All complete!")
    else:
        groups = {}
        for m in missing:
            key = (m['method'], m['dataset'], m['scoring'])
            if key not in groups:
                groups[key] = []
            groups[key].append(m['seed'])

        for (method, dataset, scoring), seeds in sorted(groups.items()):
            seeds_str = ', '.join(str(s) for s in sorted(seeds))
            print(f"  {method:>10} | {dataset:>5} | {scoring:>8} | seeds: {seeds_str}")

    total = len(missing)
    total_combos = len(set((m['method'], m['dataset'], m['scoring']) for m in missing))
    print(f"\nMissing: {total} runs ({total_combos} method-dataset-scoring combos)")
    print("=" * 60)


def run_missing(missing, ngpu=1, hidden_dims=None, output_dims=None,
                eval_csv='results/ano_results.csv',
                window_size=None, anomaly_ratio=None):
    """Run missing anomaly detection experiments."""
    # Group by (method, dataset, scoring)
    groups = {}
    for m in missing:
        key = (m['method'], m['dataset'], m['scoring'])
        if key not in groups:
            groups[key] = True

    method_order = {m: i for i, m in enumerate(METHODS)}
    sorted_groups = sorted(groups.keys(),
                           key=lambda x: (method_order.get(x[0], 999), x[1], x[2]))

    failures = []
    for method, dataset, scoring in sorted_groups:
        print(f"\n>>> {method} | {dataset} | scoring: {scoring}")

        if ngpu > 1:
            cmd = ['torchrun', f'--nproc_per_node={ngpu}', 'scripts/benchmarks/run_anomaly.py']
        else:
            cmd = [sys.executable, 'scripts/benchmarks/run_anomaly.py']

        cmd += [
            '--method', method,
            '--dataset', dataset,
            '--scoring', scoring,
            '--iterations', str(ITERATIONS),
            '--checkpoint_epochs', ','.join(str(e) for e in CHECKPOINT_EPOCHS),
        ]

        if hidden_dims is not None:
            cmd += ['--hidden_dims', str(hidden_dims)]
        if output_dims is not None:
            cmd += ['--output_dims', str(output_dims)]
        if eval_csv != 'results/ano_results.csv':
            cmd += ['--eval_csv', eval_csv]
        if dataset in DATASETS_MV:
            if window_size is not None:
                cmd += ['--window_size', str(window_size)]
            if anomaly_ratio is not None:
                cmd += ['--anomaly_ratio', str(anomaly_ratio)]

        print(f"  CMD: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failures.append((f'{method} | {dataset} | {scoring}', rc))
            print(f'  !! FAILED (exit {rc}) -- no results were written for this job')

    if failures:
        print(f'\n!! {len(failures)} job(s) failed and wrote no results:')
        for label, rc in failures:
            print(f'   {label}  (exit {rc})')
        print(f'   Child processes ran with: {sys.executable}')
        print('   A missing dependency in that interpreter is the usual cause.')


def main():
    parser = argparse.ArgumentParser(description='Check and run remaining anomaly detection experiments')
    parser.add_argument('--check', action='store_true', help='Check status only')
    parser.add_argument('--method', type=str, default=None, help='Specific method only')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs')
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
    parser.add_argument('--scoring', type=str, default=None,
                        choices=['knn', 'gaussian'],
                        help='Specific scoring method only')
    parser.add_argument('--exclude', type=str, default=None,
                        help='Comma-separated methods to exclude')
    parser.add_argument('--hidden_dims', type=int, default=None)
    parser.add_argument('--output_dims', type=int, default=None)
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of seeds (default: 3 → 42,43,44)')
    parser.add_argument('--checkpoint_epochs', type=str, default='50,100,200',
                        help='Comma-separated checkpoint epochs (default: 50,100,200). '
                             'Use "100" alone for a single-checkpoint run.')
    parser.add_argument('--eval_csv', type=str, default='results/ano_results.csv',
                        help='Output CSV path (default: results/ano_results.csv).')
    args = parser.parse_args()

    global ITERATIONS, SEEDS, CHECKPOINT_EPOCHS, MAX_EPOCH
    ITERATIONS = args.iterations
    SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51][:args.iterations]
    CHECKPOINT_EPOCHS = sorted(int(e) for e in args.checkpoint_epochs.split(',') if e.strip())
    if not CHECKPOINT_EPOCHS:
        parser.error('--checkpoint_epochs must contain at least one integer')
    MAX_EPOCH = max(CHECKPOINT_EPOCHS)

    if args.method:
        methods = [args.method]
    else:
        pool = METHODS
        methods = [m for m in pool if m not in EXCLUDED_DEFAULT]
        if args.exclude:
            exclude_set = set(args.exclude.split(','))
            methods = [m for m in methods if m not in exclude_set]

    if args.datasets:
        datasets = args.datasets
    elif args.all:
        datasets = DATASETS_UV + DATASETS_MV
    elif args.multivariate:
        datasets = DATASETS_MV
    else:
        datasets = None  # default = check_status fallback (DATASETS_UV)
    scoring_methods = [args.scoring] if args.scoring else None

    missing = check_status(methods, datasets, scoring_methods,
                           hidden_dims=args.hidden_dims, repr_dims=args.output_dims,
                           eval_csv=args.eval_csv)
    print_status(missing)

    if args.check:
        return

    if not missing:
        print("\nNothing to run!")
        return

    print(f"\n>>> Running {len(missing)} missing anomaly detection jobs...")
    run_missing(missing, args.ngpu,
                hidden_dims=args.hidden_dims, output_dims=args.output_dims,
                eval_csv=args.eval_csv,
                window_size=args.window_size, anomaly_ratio=args.anomaly_ratio)

    # Re-check
    print("\n\n>>> Final status:")
    missing = check_status(methods, datasets, scoring_methods,
                           hidden_dims=args.hidden_dims, repr_dims=args.output_dims,
                           eval_csv=args.eval_csv)
    print_status(missing)


if __name__ == '__main__':
    main()
