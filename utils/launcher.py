"""Shared pieces of the launchers in the repository root (run_*.py).

Each launcher checks the result CSV for missing runs, prints a status table,
and starts the benchmark runner in scripts/benchmarks/ for what is missing.
Kept free of torch so `--check` stays fast.
"""

import subprocess
import sys

from utils.result_status import default_seeds


def add_common_args(parser, *, checkpoint_default, eval_csv_default, eval_csv_help):
    """Arguments every launcher takes. Task-specific ones are added by the caller."""
    parser.add_argument('--check', action='store_true', help='Check status only')
    parser.add_argument('--method', type=str, default=None, help='Specific method only')
    parser.add_argument('--exclude', type=str, default=None,
                        help='Comma-separated methods to exclude (e.g. ts2vec,timesurl)')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--hidden_dims', type=int, default=None,
                        help='Override hidden dims (default: per-method)')
    parser.add_argument('--output_dims', type=int, default=None,
                        help='Override output/repr dims (default: per-method)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of seeds (default: 3, i.e. 42,43,44)')
    parser.add_argument('--checkpoint_epochs', type=str, default=checkpoint_default,
                        help=f'Comma-separated checkpoint epochs (default: {checkpoint_default}). '
                             'Use "100" alone for a single-checkpoint run.')
    parser.add_argument('--eval_csv', type=str, default=eval_csv_default, help=eval_csv_help)


def parse_run_plan(parser, args):
    """(seeds, sorted checkpoint epochs) from the parsed arguments."""
    checkpoint_epochs = sorted(int(e) for e in args.checkpoint_epochs.split(',') if e.strip())
    if not checkpoint_epochs:
        parser.error('--checkpoint_epochs must contain at least one integer')
    return default_seeds(args.iterations), checkpoint_epochs


def select_methods(args, methods, always_excluded=()):
    """--method, else `methods` minus --exclude."""
    if args.method:
        return [args.method]
    excluded = set(always_excluded)
    if args.exclude:
        excluded |= {m.strip() for m in args.exclude.split(',')}
    return [m for m in methods if m not in excluded]


def print_seed_status(title, header_lines, missing, keys, combo_label):
    """Status table: one line per `keys` combination (3 keys) listing its missing seeds."""
    print("\n" + "=" * 60)
    print(f" {title}")
    for line in header_lines:
        print(f" {line}")
    print("=" * 60)

    groups = {}
    for m in missing:
        groups.setdefault(tuple(m[k] for k in keys), []).append(m['seed'])
    if not groups:
        print("\n  All complete!")
    for (a, b, c), seeds in sorted(groups.items()):
        print(f"  {a:>10} | {b:>5} | {c:>8} | seeds: {', '.join(str(s) for s in sorted(seeds))}")

    print(f"\nMissing: {len(missing)} runs ({len(groups)} {combo_label} combos)")
    print("=" * 60)


def run_jobs(jobs, script, ngpu=1):
    """Run each (label, args) job as `script args`, under torchrun when ngpu > 1.

    A failed job does not stop the rest; the failures are listed at the end.
    """
    failures = []
    for label, args in jobs:
        print(f"\n>>> {label}")
        if ngpu > 1:
            cmd = ['torchrun', f'--nproc_per_node={ngpu}', script]
        else:
            cmd = [sys.executable, script]
        cmd += args
        print(f"  CMD: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            failures.append((label, rc))
            print(f'  !! FAILED (exit {rc}) -- see the error above')

    if failures:
        print(f'\n!! {len(failures)} job(s) failed:')
        for label, rc in failures:
            print(f'   {label}  (exit {rc})')
        print(f'   Child processes ran with: {sys.executable}')
        print('   Results of the jobs that did finish were written; '
              'rerun to retry the rest.')
    return failures
