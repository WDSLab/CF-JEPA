"""
Check and run remaining forecast experiments (checkpoint mode).

Checks fcst_results.csv and only runs what's missing.

Usage:
    python run_forecast.py --check                              # Check status only
    python run_forecast.py                                      # Run all missing
    python run_forecast.py --method ts2vec                      # Specific method
    python run_forecast.py --method ts2vec --ngpu 2             # Multi-GPU
    python run_forecast.py --univar_only                        # Univariate only
    python run_forecast.py --multivar_only                      # Multivariate only
    python run_forecast.py --datasets ETTh1 ETTh2               # Specific datasets
    python run_forecast.py --exclude timesurl,simmtm --ngpu 2  # Exclude methods
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
    'infots', 'cost', 'ts_jepa', 'fei', 'autotcl', 'cf_jepa', 'timesurl',
]

# Paper scope: 8 datasets (electricity excluded — out of paper scope)
DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'AirQuality', 'SML', 'Traffic']
CHECKPOINT_EPOCHS = [50, 100, 200, 400]
MAX_EPOCH = 400
ITERATIONS = 3
SEEDS = [42, 43, 44]


def _is_done(method, dataset, seed, max_epoch, univar, hidden_dims=None, repr_dims=None,
             eval_csv='results/fcst_results.csv'):
    """Check if last checkpoint results exist in forecast CSV."""
    if not os.path.exists(eval_csv):
        return False
    univar_str = str(univar)
    with open(eval_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get('method') == method and
                row.get('dataset') == dataset and
                row.get('seed') == str(seed) and
                row.get('epoch') == str(max_epoch) and
                row.get('hidden_dims', str(hidden_dims)) == str(hidden_dims) and
                row.get('repr_dims', str(repr_dims)) == str(repr_dims) and
                row.get('univar') == univar_str):
                return True
    return False


def check_status(methods=None, datasets=None, modes=None,
                 hidden_dims=None, repr_dims=None,
                 eval_csv='results/fcst_results.csv'):
    """Check which (method, dataset, mode, seed) combos are incomplete."""
    methods = methods or METHODS
    datasets = datasets or DATASETS
    if modes is None:
        modes = [False, True]  # multivar, univar

    missing = []
    for method in methods:
        default_h, default_r = get_method_dims(method)
        actual_h = hidden_dims if hidden_dims is not None else default_h
        actual_r = repr_dims if repr_dims is not None else default_r
        for dataset in datasets:
            for univar in modes:
                for seed in SEEDS:
                    if not _is_done(method, dataset, seed, MAX_EPOCH, univar,
                                    hidden_dims=actual_h, repr_dims=actual_r,
                                    eval_csv=eval_csv):
                        missing.append({
                            'method': method,
                            'dataset': dataset,
                            'univar': univar,
                            'seed': seed,
                        })
    return missing


def print_status(missing):
    print("\n" + "=" * 60)
    print(" Forecast Experiment Status")
    print(f" Checkpoints: {CHECKPOINT_EPOCHS}")
    print(f" Seeds: {SEEDS}")
    print("=" * 60)

    if not missing:
        print("\n  All complete!")
    else:
        groups = {}
        for m in missing:
            mode_str = 'univar' if m['univar'] else 'multivar'
            key = (m['method'], m['dataset'], mode_str)
            if key not in groups:
                groups[key] = []
            groups[key].append(m['seed'])

        for (method, dataset, mode), seeds in sorted(groups.items()):
            seeds_str = ', '.join(str(s) for s in sorted(seeds))
            print(f"  {method:>10} | {dataset:>5} | {mode:>8} | seeds: {seeds_str}")

    total = len(missing)
    total_combos = len(set((m['method'], m['dataset'], m['univar']) for m in missing))
    print(f"\nMissing: {total} runs ({total_combos} method-dataset-mode combos)")
    print("=" * 60)


def run_missing(missing, ngpu=1, hidden_dims=None, output_dims=None,
                eval_csv='results/fcst_results.csv'):
    """Run missing forecast experiments."""
    # Group by (method, univar) — each run handles one method + mode
    groups = {}
    for m in missing:
        key = (m['method'], m['univar'])
        if key not in groups:
            groups[key] = set()
        groups[key].add(m['dataset'])

    method_order = {m: i for i, m in enumerate(METHODS)}
    sorted_groups = sorted(groups.items(),
                           key=lambda x: (method_order.get(x[0][0], 999), x[0][1]))

    failures = []
    for (method, univar), datasets in sorted_groups:
        mode_str = 'univar' if univar else 'multivar'
        print(f"\n>>> {method} | {mode_str} | datasets: {sorted(datasets)}")

        if ngpu > 1:
            cmd = ['torchrun', f'--nproc_per_node={ngpu}', 'scripts/benchmarks/run_forecast.py']
        else:
            cmd = [sys.executable, 'scripts/benchmarks/run_forecast.py']

        cmd += [
            '--method', method,
            '--iterations', str(ITERATIONS),
            '--checkpoint_epochs', ','.join(str(e) for e in CHECKPOINT_EPOCHS),
        ]

        # Add datasets
        if set(datasets) == set(DATASETS):
            cmd.append('--all_ett')
        else:
            for ds in sorted(datasets):
                cmd += ['--dataset', ds]

        if univar:
            cmd.append('--univar')

        if hidden_dims is not None:
            cmd += ['--hidden_dims', str(hidden_dims)]
        if output_dims is not None:
            cmd += ['--output_dims', str(output_dims)]
        if eval_csv != 'results/fcst_results.csv':
            cmd += ['--eval_csv', eval_csv]

        print(f"  CMD: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failures.append((f'{method} | {mode_str}', rc))
            print(f'  !! FAILED (exit {rc}) -- no results were written for this job')

    if failures:
        print(f'\n!! {len(failures)} job(s) failed and wrote no results:')
        for label, rc in failures:
            print(f'   {label}  (exit {rc})')
        print(f'   Child processes ran with: {sys.executable}')
        print('   A missing dependency in that interpreter is the usual cause.')


def main():
    parser = argparse.ArgumentParser(description='Check and run remaining forecast experiments')
    parser.add_argument('--check', action='store_true', help='Check status only')
    parser.add_argument('--method', type=str, default=None, help='Specific method only')
    parser.add_argument('--exclude', type=str, default=None,
                        help='Comma-separated methods to exclude (e.g. timesurl,simmtm)')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--univar_only', action='store_true', help='Univariate only')
    parser.add_argument('--multivar_only', action='store_true', help='Multivariate only')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets (e.g., ETTh1 ETTh2)')
    parser.add_argument('--hidden_dims', type=int, default=None,
                        help='Override hidden dims (default: per-method)')
    parser.add_argument('--output_dims', type=int, default=None,
                        help='Override output/repr dims (default: per-method)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of seeds (default: 3 → 42,43,44)')
    parser.add_argument('--checkpoint_epochs', type=str, default='50,100,200,400',
                        help='Comma-separated checkpoint epochs (default: 50,100,200,400). '
                             'Use "100" alone for a single-checkpoint run.')
    parser.add_argument('--eval_csv', type=str, default='results/fcst_results.csv',
                        help='Output CSV path (default: results/fcst_results.csv). '
                             'Use a separate path for verification runs.')
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
        methods = list(METHODS)
        if args.exclude:
            excluded = [m.strip() for m in args.exclude.split(',')]
            methods = [m for m in methods if m not in excluded]
    datasets = args.datasets
    # Paper scope: electricity is out of scope — drop unconditionally.
    if datasets and 'electricity' in datasets:
        print("[WARN] 'electricity' is out of paper scope — excluding from run")
        datasets = [d for d in datasets if d != 'electricity']

    modes = None
    if args.univar_only:
        modes = [True]
    elif args.multivar_only:
        modes = [False]

    missing = check_status(methods, datasets, modes,
                           hidden_dims=args.hidden_dims, repr_dims=args.output_dims,
                           eval_csv=args.eval_csv)
    print_status(missing)

    if args.check:
        return

    if not missing:
        print("\nNothing to run!")
        return

    print(f"\n>>> Running {len(missing)} missing forecast jobs...")
    run_missing(missing, args.ngpu,
                hidden_dims=args.hidden_dims, output_dims=args.output_dims,
                eval_csv=args.eval_csv)

    # Re-check
    print("\n\n>>> Final status:")
    missing = check_status(methods, datasets, modes,
                           hidden_dims=args.hidden_dims, repr_dims=args.output_dims,
                           eval_csv=args.eval_csv)
    print_status(missing)


if __name__ == '__main__':
    main()
