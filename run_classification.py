"""
Check and run remaining checkpoint experiments (train + save representations).

Checks for existing raw.npz files and only runs what's missing.

Usage:
    python run_checkpoints.py --check                           # Check status only
    python run_checkpoints.py --task classification             # Run missing UCR+UEA
    python run_checkpoints.py --task classification --ucr_only  # UCR only
    python run_checkpoints.py --task classification --uea_only  # UEA only
    python run_checkpoints.py --method ts2vec                   # Specific method
    python run_checkpoints.py --task classification --ngpu 2    # Multi-GPU
    python run_checkpoints.py --check --method ts2vec           # Check specific method
"""

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import get_method_dims

# Paper scope: UCR 126 (t1+t2+t3) + UEA 26 (t1+t2). t4/t3 excluded.
UCR_SPLIT_FILES = [
    'splits/datasets_ucr_t1.txt', 'splits/datasets_ucr_t2.txt',
    'splits/datasets_ucr_t3.txt',
]
UEA_SPLIT_FILES = [
    'splits/datasets_uea_t1.txt', 'splits/datasets_uea_t2.txt',
]


def _build_default_datasets_file(data_type):
    """Concatenate splits/datasets_{ucr,uea}_t*.txt into a temp file. Cached per data_type."""
    if data_type == 'UCR':
        paths = UCR_SPLIT_FILES
    elif data_type == 'UEA':
        paths = UEA_SPLIT_FILES
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    datasets = []
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                datasets.extend(line.strip() for line in f if line.strip())
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix=f'_{data_type.lower()}.txt',
                                      delete=False, encoding='utf-8')
    tmp.write('\n'.join(datasets) + '\n')
    tmp.close()
    return tmp.name

METHODS = [
    'ts2vec', 'softclt', 'trep',
    'infots', 'cost', 'simmtm', 'ts_jepa', 'fei', 'autotcl', 'cf_jepa', 'timesurl',
]

CHECKPOINT_EPOCHS = [50, 100, 200]
MAX_EPOCH = 200
ITERATIONS = 3
SEEDS = [42, 43, 44]


def _load_datasets_file(path):
    """Load dataset names from text file (one per line)."""
    datasets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                datasets.append(line)
    return datasets


def _load_completed_from_csv(eval_csv, method, seed, max_epoch, hidden_dims=None, repr_dims=None):
    """Return set of dataset names that have ALL results for given method, seed.

    Each dataset needs 4 rows (2 pooling × 2 eval_protocol) at EVERY checkpoint epoch.
    Total expected: len(CHECKPOINT_EPOCHS) × 4 rows per dataset.
    """
    completed = set()
    if not os.path.exists(eval_csv):
        return completed
    if hidden_dims is None:
        hidden_dims, _ = get_method_dims(method)
    if repr_dims is None:
        _, repr_dims = get_method_dims(method)
    import csv
    # Count rows per (dataset, epoch)
    dataset_epoch_counts = {}
    with open(eval_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get('method') == method and
                row.get('seed') == str(seed) and
                row.get('hidden_dims', '64') == str(hidden_dims) and
                row.get('repr_dims', '320') == str(repr_dims)):
                ds = row.get('dataset')
                ep = row.get('epoch')
                key = (ds, ep)
                dataset_epoch_counts[key] = dataset_epoch_counts.get(key, 0) + 1
    # A dataset is complete only if ALL checkpoint epochs have 4 rows each
    all_datasets = set(ds for ds, _ in dataset_epoch_counts.keys())
    for ds in all_datasets:
        all_complete = True
        for ep in CHECKPOINT_EPOCHS:
            if dataset_epoch_counts.get((ds, str(ep)), 0) < 4:
                all_complete = False
                break
        if all_complete:
            completed.add(ds)
    return completed


def check_checkpoint_status(methods=None, data_types=None, datasets_file=None, hidden_dims=None, output_dims=None, eval_csv='results/cls_results.csv'):
    """Check which (method, data_type, seed) combos are incomplete.

    A combo is complete when cls_results.csv has entries for the last checkpoint epoch.
    """
    methods = methods or METHODS
    if data_types is None:
        data_types = ['UCR', 'UEA']

    # UCR datasets are in data/UCR/, UEA datasets are in data/UEA/
    ucr_path = os.path.join('data', 'UCR')
    uea_path = os.path.join('data', 'UEA')
    ucr_datasets = set(os.listdir(ucr_path)) if os.path.exists(ucr_path) else set()
    uea_datasets = set(os.listdir(uea_path)) if os.path.exists(uea_path) else set()
    type_filter = {'UCR': ucr_datasets, 'UEA': uea_datasets}

    if datasets_file:
        custom_datasets = _load_datasets_file(datasets_file)
        expected_counts = {}
        for dt in data_types:
            if dt in type_filter:
                expected_counts[dt] = len([d for d in custom_datasets if d in type_filter[dt]])
            else:
                expected_counts[dt] = len(custom_datasets)
    else:
        custom_datasets = None
        # Paper scope: UCR 126 (t1+t2+t3) + UEA 26 (t1+t2). Must match UCR_SPLIT_FILES / UEA_SPLIT_FILES above.
        expected_counts = {'UCR': 126, 'UEA': 26}

    missing = []

    for method in methods:
        for data_type in data_types:
            n_expected = expected_counts.get(data_type, 0)

            for seed in SEEDS:
                h = hidden_dims if hidden_dims is not None else get_method_dims(method)[0]
                r = output_dims if output_dims is not None else get_method_dims(method)[1]
                completed = _load_completed_from_csv(eval_csv, method, seed, MAX_EPOCH, hidden_dims=h, repr_dims=r)
                # Filter completed to only datasets of this data_type (fixes prior bug
                # where len(completed) included UCR+UEA together when no datasets_file
                # was given — UEA was wrongly marked done as long as UCR was complete).
                if custom_datasets:
                    done = len(completed & set(custom_datasets) & type_filter.get(data_type, set()))
                elif data_type in type_filter:
                    done = len(completed & type_filter[data_type])
                else:
                    done = len(completed)

                if done < n_expected:
                    missing.append({
                        'method': method,
                        'data_type': data_type,
                        'seed': seed,
                        'done': done,
                        'expected': n_expected,
                    })

    return missing


def print_status(missing):
    print("\n" + "=" * 60)
    print(" Checkpoint Experiment Status")
    print(f" Checkpoints: {CHECKPOINT_EPOCHS}")
    print(f" Seeds: {SEEDS}")
    print("=" * 60)

    if not missing:
        print("\n  All complete!")
    else:
        # Group by (method, data_type)
        groups = {}
        for m in missing:
            key = (m['method'], m['data_type'])
            if key not in groups:
                groups[key] = []
            groups[key].append(m)

        for (method, data_type), items in sorted(groups.items()):
            seeds_info = ', '.join(
                f"seed{m['seed']}({m['done']}/{m['expected']})" for m in items
            )
            print(f"  {method:>10} | {data_type:>3} | {seeds_info}")

    total = len(missing)
    print(f"\nTotal missing (method, data_type, seed) combos: {total}")
    print("=" * 60)


def run_missing(missing, ngpu=1, datasets_file=None, hidden_dims=None, output_dims=None, eval_csv='results/cls_results.csv'):
    """Run missing checkpoint experiments."""
    import subprocess

    # Group by (method, data_type) — each run handles all seeds via --iterations
    groups = {}
    for m in missing:
        key = (m['method'], m['data_type'])
        if key not in groups:
            groups[key] = set()
        groups[key].add(m['seed'])

    # Sort: UCR first, then UEA; autotcl always last
    order_dt = {'UCR': 0, 'UEA': 1}
    method_order = {m: i for i, m in enumerate(METHODS)}
    sorted_groups = sorted(groups.items(),
                           key=lambda x: (order_dt.get(x[0][1], 2), method_order.get(x[0][0], 999)))

    default_files = {}  # cache temp files per data_type when datasets_file is None

    for (method, data_type), seeds_needed in sorted_groups:
        n_iters = ITERATIONS

        print(f"\n>>> {method} | {data_type} | seeds: {sorted(seeds_needed)}")

        if ngpu > 1:
            cmd = ['torchrun', f'--nproc_per_node={ngpu}', 'scripts/benchmarks/run_classification.py']
        else:
            cmd = ['python', 'scripts/benchmarks/run_classification.py']

        cmd += [
            '--method', method,
            '--iterations', str(n_iters),
            '--checkpoint_mode',
            '--checkpoint_epochs', ','.join(str(e) for e in CHECKPOINT_EPOCHS),
        ]

        if datasets_file:
            cmd += ['--datasets_file', datasets_file]
        else:
            if data_type not in default_files:
                default_files[data_type] = _build_default_datasets_file(data_type)
            cmd += ['--datasets_file', default_files[data_type]]

        if hidden_dims is not None:
            cmd += ['--hidden_dims', str(hidden_dims)]
        if output_dims is not None:
            cmd += ['--output_dims', str(output_dims)]
        if eval_csv != 'results/cls_results.csv':
            cmd += ['--eval_csv', eval_csv]

        print(f"  CMD: {' '.join(cmd)}")
        subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description='Check and run remaining checkpoint experiments')
    parser.add_argument('--check', action='store_true', help='Check status only')
    parser.add_argument('--task', type=str, default='classification', choices=['classification'])
    parser.add_argument('--method', type=str, default=None, help='Specific method only')
    parser.add_argument('--exclude', type=str, default=None,
                        help='Comma-separated methods to exclude (e.g. ts2vec,softclt,timesurl)')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--ucr_only', action='store_true', help='UCR only')
    parser.add_argument('--uea_only', action='store_true', help='UEA only')
    parser.add_argument('--datasets_file', type=str, default=None,
                        help='Text file with dataset names (one per line)')
    parser.add_argument('--hidden_dims', type=int, default=None,
                        help='Override hidden dims')
    parser.add_argument('--output_dims', type=int, default=None,
                        help='Override output/repr dims')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of seeds (default: 3 → 42,43,44)')
    parser.add_argument('--checkpoint_epochs', type=str, default='50,100,200',
                        help='Comma-separated checkpoint epochs (default: 50,100,200). '
                             'Use "100" alone for a single-checkpoint run.')
    parser.add_argument('--eval_csv', type=str, default='results/cls_results.csv',
                        help='Output CSV path (default: results/cls_results.csv). '
                             'Use a separate path for verification runs to keep main results clean.')
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

    data_types = None
    if args.ucr_only:
        data_types = ['UCR']
    elif args.uea_only:
        data_types = ['UEA']

    missing = check_checkpoint_status(methods, data_types, args.datasets_file,
                                      hidden_dims=args.hidden_dims, output_dims=args.output_dims,
                                      eval_csv=args.eval_csv)
    print_status(missing)

    if args.check:
        return

    if not missing:
        print("\nNothing to run!")
        return

    print(f"\n>>> Running {len(missing)} missing checkpoint jobs...")
    run_missing(missing, args.ngpu, args.datasets_file,
                hidden_dims=args.hidden_dims, output_dims=args.output_dims,
                eval_csv=args.eval_csv)

    # Re-check
    print("\n\n>>> Final status:")
    missing = check_checkpoint_status(methods, data_types, args.datasets_file,
                                      hidden_dims=args.hidden_dims, output_dims=args.output_dims,
                                      eval_csv=args.eval_csv)
    print_status(missing)


if __name__ == '__main__':
    main()
