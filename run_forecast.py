"""
Check and run remaining forecast experiments.

Checks results/fcst_results.csv and only runs what's missing.

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
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.launcher import (
    add_common_args, parse_run_plan, print_seed_status, run_jobs, select_methods,
)
from utils.result_status import FCST_CSV, dims_for, fcst_done

METHODS = [
    'ts2vec', 'softclt', 'trep',
    'infots', 'cost', 'ts_jepa', 'fei', 'autotcl', 'cf_jepa', 'timesurl',
]
# Paper scope (electricity excluded); same list as the runner's FORECAST_DATASETS
DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'AirQuality', 'SML', 'Traffic']
RUNNER = 'scripts/benchmarks/run_forecast.py'


def check_status(methods, datasets, modes, seeds, max_epoch,
                 hidden_dims=None, output_dims=None, eval_csv=FCST_CSV):
    """Which (method, dataset, mode, seed) combos are incomplete."""
    missing = []
    for method in methods:
        h, r = dims_for(method, hidden_dims, output_dims)
        for dataset in datasets:
            for univar in modes:
                for seed in seeds:
                    if not fcst_done(eval_csv, method, dataset, seed, max_epoch, univar, h, r):
                        missing.append({'method': method, 'dataset': dataset,
                                        'univar': univar, 'seed': seed})
    return missing


def run_missing(missing, args, checkpoint_epochs):
    """Run missing experiments, one job per (method, mode)."""
    groups = {}
    for m in missing:
        groups.setdefault((m['method'], m['univar']), set()).add(m['dataset'])
    method_order = {m: i for i, m in enumerate(METHODS)}

    jobs = []
    for (method, univar), datasets in sorted(groups.items(),
                                             key=lambda x: (method_order.get(x[0][0], 999), x[0][1])):
        cmd = ['--method', method, '--iterations', str(args.iterations),
               '--checkpoint_epochs', ','.join(str(e) for e in checkpoint_epochs)]
        if set(datasets) == set(DATASETS):
            cmd.append('--all_datasets')
        else:
            for ds in sorted(datasets):
                cmd += ['--dataset', ds]
        if univar:
            cmd.append('--univar')
        for flag, value in [('--hidden_dims', args.hidden_dims), ('--output_dims', args.output_dims)]:
            if value is not None:
                cmd += [flag, str(value)]
        if args.eval_csv != FCST_CSV:
            cmd += ['--eval_csv', args.eval_csv]
        mode = 'univar' if univar else 'multivar'
        jobs.append((f'{method} | {mode} | datasets: {sorted(datasets)}', cmd))

    run_jobs(jobs, RUNNER, args.ngpu)


def main():
    parser = argparse.ArgumentParser(description='Check and run remaining forecast experiments')
    add_common_args(parser, checkpoint_default='50,100,200,400', eval_csv_default=FCST_CSV,
                    eval_csv_help=f'Output CSV path (default: {FCST_CSV}). '
                                  'Use a separate path for verification runs.')
    parser.add_argument('--univar_only', action='store_true', help='Univariate only')
    parser.add_argument('--multivar_only', action='store_true', help='Multivariate only')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets (e.g., ETTh1 ETTh2)')
    args = parser.parse_args()

    seeds, checkpoint_epochs = parse_run_plan(parser, args)
    methods = select_methods(args, METHODS)

    datasets = args.datasets
    if datasets and 'electricity' in datasets:
        print("[WARN] 'electricity' is out of paper scope - excluding from run")
        datasets = [d for d in datasets if d != 'electricity']
    datasets = datasets or DATASETS

    if args.univar_only:
        modes = [True]
    elif args.multivar_only:
        modes = [False]
    else:
        modes = [False, True]  # multivar, univar

    def check():
        return check_status(methods, datasets, modes, seeds, max(checkpoint_epochs),
                            args.hidden_dims, args.output_dims, args.eval_csv)

    def show(missing):
        rows = [dict(m, mode='univar' if m['univar'] else 'multivar') for m in missing]
        print_seed_status('Forecast Experiment Status',
                          [f'Checkpoints: {checkpoint_epochs}', f'Seeds: {seeds}'],
                          rows, keys=('method', 'dataset', 'mode'),
                          combo_label='method-dataset-mode')

    missing = check()
    show(missing)
    if args.check:
        return
    if not missing:
        print("\nNothing to run!")
        return
    print(f"\n>>> Running {len(missing)} missing forecast jobs...")
    run_missing(missing, args, checkpoint_epochs)
    print("\n\n>>> Final status:")
    show(check())


if __name__ == '__main__':
    main()
