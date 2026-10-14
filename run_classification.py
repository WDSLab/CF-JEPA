"""
Check and run remaining classification experiments.

Checks results/cls_results.csv and only runs what's missing.

Usage:
    python run_classification.py --check                        # Check status only
    python run_classification.py                                # Run missing UCR+UEA
    python run_classification.py --ucr_only                     # UCR only
    python run_classification.py --uea_only                     # UEA only
    python run_classification.py --method ts2vec                # Specific method
    python run_classification.py --ngpu 2                       # Multi-GPU
    python run_classification.py --datasets_file splits/datasets_ucr_t1.txt  # Subset
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import UCR_PATH, UEA_PATH
from utils.launcher import add_common_args, parse_run_plan, run_jobs, select_methods
from utils.result_status import CLS_CSV, cls_completed_datasets, dims_for

# Paper scope: UCR 126 (t1+t2+t3) + UEA 26 (t1+t2). t4/t3 excluded.
SPLIT_FILES = {
    'UCR': ['splits/datasets_ucr_t1.txt', 'splits/datasets_ucr_t2.txt',
            'splits/datasets_ucr_t3.txt'],
    'UEA': ['splits/datasets_uea_t1.txt', 'splits/datasets_uea_t2.txt'],
}
PAPER_SCOPE_COUNTS = {'UCR': 126, 'UEA': 26}  # must match SPLIT_FILES

METHODS = [
    'ts2vec', 'softclt', 'trep',
    'infots', 'cost', 'simmtm', 'ts_jepa', 'fei', 'autotcl', 'cf_jepa', 'timesurl',
]
RUNNER = 'scripts/benchmarks/run_classification.py'


def _read_lines(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _paper_scope_file(data_type):
    """The split files of one archive concatenated into a temp file."""
    datasets = []
    for p in SPLIT_FILES[data_type]:
        if os.path.exists(p):
            datasets.extend(_read_lines(p))
    with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{data_type.lower()}.txt',
                                     delete=False, encoding='utf-8') as tmp:
        tmp.write('\n'.join(datasets) + '\n')
    return tmp.name


def check_status(methods, data_types, datasets_file, seeds, checkpoint_epochs,
                 hidden_dims=None, output_dims=None, eval_csv=CLS_CSV):
    """Which (method, data_type, seed) combos still miss datasets."""
    on_disk = {dt: set(os.listdir(path)) if os.path.exists(path) else set()
               for dt, path in (('UCR', UCR_PATH), ('UEA', UEA_PATH))}
    if datasets_file:
        custom = set(_read_lines(datasets_file))
        expected = {dt: len(custom & on_disk[dt]) for dt in data_types}
    else:
        custom = None
        expected = PAPER_SCOPE_COUNTS

    missing = []
    for method in methods:
        h, r = dims_for(method, hidden_dims, output_dims)
        for data_type in data_types:
            for seed in seeds:
                # UCR and UEA share the CSV, so count only this archive's datasets.
                done = cls_completed_datasets(eval_csv, method, seed, checkpoint_epochs, h, r)
                done &= on_disk[data_type]
                if custom:
                    done &= custom
                if len(done) < expected.get(data_type, 0):
                    missing.append({'method': method, 'data_type': data_type, 'seed': seed,
                                    'done': len(done), 'expected': expected[data_type]})
    return missing


def print_status(missing, checkpoint_epochs, seeds):
    print("\n" + "=" * 60)
    print(" Checkpoint Experiment Status")
    print(f" Checkpoints: {checkpoint_epochs}")
    print(f" Seeds: {seeds}")
    print("=" * 60)

    groups = {}
    for m in missing:
        groups.setdefault((m['method'], m['data_type']), []).append(m)
    if not groups:
        print("\n  All complete!")
    for (method, data_type), items in sorted(groups.items()):
        seeds_info = ', '.join(f"seed{m['seed']}({m['done']}/{m['expected']})" for m in items)
        print(f"  {method:>10} | {data_type:>3} | {seeds_info}")

    print(f"\nTotal missing (method, data_type, seed) combos: {len(missing)}")
    print("=" * 60)


def run_missing(missing, args, checkpoint_epochs):
    """Run missing experiments, one job per (method, data_type); each covers all seeds."""
    groups = {}
    for m in missing:
        groups.setdefault((m['method'], m['data_type']), set()).add(m['seed'])
    # UCR first, then UEA; within each, METHODS order (slow ones last)
    order_dt = {'UCR': 0, 'UEA': 1}
    method_order = {m: i for i, m in enumerate(METHODS)}

    scope_files = {}
    jobs = []
    for (method, data_type), seeds_needed in sorted(
            groups.items(), key=lambda x: (order_dt.get(x[0][1], 2), method_order.get(x[0][0], 999))):
        if not args.datasets_file and data_type not in scope_files:
            scope_files[data_type] = _paper_scope_file(data_type)
        cmd = ['--method', method, '--iterations', str(args.iterations),
               '--checkpoint_epochs', ','.join(str(e) for e in checkpoint_epochs),
               '--datasets_file', args.datasets_file or scope_files[data_type]]
        for flag, value in [('--hidden_dims', args.hidden_dims), ('--output_dims', args.output_dims)]:
            if value is not None:
                cmd += [flag, str(value)]
        if args.eval_csv != CLS_CSV:
            cmd += ['--eval_csv', args.eval_csv]
        jobs.append((f'{method} | {data_type} | seeds: {sorted(seeds_needed)}', cmd))

    run_jobs(jobs, RUNNER, args.ngpu)


def main():
    parser = argparse.ArgumentParser(description='Check and run remaining classification experiments')
    add_common_args(parser, checkpoint_default='50,100,200', eval_csv_default=CLS_CSV,
                    eval_csv_help=f'Output CSV path (default: {CLS_CSV}). Use a separate '
                                  'path for verification runs to keep main results clean.')
    parser.add_argument('--ucr_only', action='store_true', help='UCR only')
    parser.add_argument('--uea_only', action='store_true', help='UEA only')
    parser.add_argument('--datasets_file', type=str, default=None,
                        help='Text file with dataset names (one per line)')
    args = parser.parse_args()

    seeds, checkpoint_epochs = parse_run_plan(parser, args)
    methods = select_methods(args, METHODS)
    if args.ucr_only:
        data_types = ['UCR']
    elif args.uea_only:
        data_types = ['UEA']
    else:
        data_types = ['UCR', 'UEA']

    def check():
        return check_status(methods, data_types, args.datasets_file, seeds, checkpoint_epochs,
                            args.hidden_dims, args.output_dims, args.eval_csv)

    missing = check()
    print_status(missing, checkpoint_epochs, seeds)
    if args.check:
        return
    if not missing:
        print("\nNothing to run!")
        return
    print(f"\n>>> Running {len(missing)} missing checkpoint jobs...")
    run_missing(missing, args, checkpoint_epochs)
    print("\n\n>>> Final status:")
    print_status(check(), checkpoint_epochs, seeds)


if __name__ == '__main__':
    main()
