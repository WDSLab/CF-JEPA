"""Plumbing shared by the benchmark runners in scripts/benchmarks/.

DDP setup, seeding, the train-once/evaluate-at-checkpoints loop, the
multi-GPU dataset queue, output files and failure tracking. Encoding and
evaluation stay in each runner.
"""

import contextlib
import csv
import errno
import gc
import os
import random
import sys
import tempfile
import time
import traceback
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist


def setup_ddp():
    """Join the process group when launched by torchrun.

    Returns (rank, world_size, local_rank); (0, 1, 0) for a plain launch.
    """
    if 'RANK' not in os.environ:
        return 0, 1, 0
    dist.init_process_group(backend='gloo', timeout=timedelta(hours=24))
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def main_print(*args, **kwargs):
    """print() on rank 0 only (always, for a plain launch)."""
    if is_main_process():
        print(*args, **kwargs)


def banner(*lines):
    main_print(f"\n{'=' * 60}")
    for line in lines:
        main_print(line)
    main_print('=' * 60)


def get_device(local_rank):
    return f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_epochs(text):
    """'200,50,100' -> [50, 100, 200]."""
    return sorted({int(x) for x in text.split(',') if x.strip()})


def auto_max_train_length(method_kwargs, T, limit=2048):
    """Cap the training crop length for long sequences unless set explicitly."""
    if 'max_train_length' not in method_kwargs and T > limit:
        method_kwargs['max_train_length'] = limit
        main_print(f"  Auto max_train_length={limit} (T={T})")


def dual_encoder_ready(trainer):
    """True when the trainer exposes an EMA target encoder (CF-JEPA family)."""
    return getattr(trainer, 'target_encoder', None) is not None


def train_with_checkpoints(trainer, train_data, checkpoint_epochs, evaluate, loss_csv):
    """Fit once for max(checkpoint_epochs) epochs, calling evaluate(trainer, epoch)
    at each checkpoint, then write the per-epoch loss log.

    One continuous fit() is required: several upstream fit() implementations
    (TS2Vec, SoftCLT, TimesURL, CoST) create a fresh optimizer per call, so
    fitting in chunks would reset optimizer and scheduler state.
    Returns the wall-clock training time in seconds.
    """
    checkpoints = set(checkpoint_epochs)
    loss_log = []

    def callback(tr, epoch, loss):
        loss_log.append((epoch, loss))
        if epoch in checkpoints:
            evaluate(tr, epoch)
            gc.collect()

    main_print(f"\n  Training 1 -> {max(checkpoints)} (continuous)...")
    t_start = time.time()
    trainer.fit(train_data, epochs=max(checkpoints), verbose=False, callback=callback)
    train_time = time.time() - t_start

    os.makedirs(os.path.dirname(loss_csv), exist_ok=True)
    with open(loss_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss'])
        writer.writeheader()
        for epoch, loss in loss_log:
            writer.writerow({'epoch': epoch, 'loss': round(loss, 6)})
    main_print(f"\n  Loss log: {loss_csv}")
    return train_time


def append_csv_rows(path, fieldnames, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


@contextlib.contextmanager
def _exclusive_lock(f):
    if os.name == 'nt':
        import msvcrt
        f.seek(0)
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as e:  # held by another rank
                if e.errno not in (errno.EACCES, errno.EDEADLOCK):
                    raise
                time.sleep(0.05)
        try:
            yield
        finally:
            f.seek(0)  # flushes the pending write before the byte is unlocked
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def iter_work(items, rank, world_size, tag):
    """Yield (index, item) pairs for this process.

    A plain launch gets every item in order. Under torchrun the ranks share a
    counter file and each takes the next unclaimed item, so one slow dataset
    does not hold up the other GPUs. Every rank must exhaust the generator:
    it starts and ends on a barrier.
    """
    if world_size == 1:
        yield from enumerate(items)
        return

    counter = os.path.join(tempfile.gettempdir(),
                           f'ssl_{tag}_{os.environ.get("MASTER_PORT", "29500")}.counter')
    if rank == 0:
        with open(counter, 'w') as f:
            f.write('0')
    dist.barrier()
    while True:
        with open(counter, 'r+') as f, _exclusive_lock(f):
            idx = int(f.read().strip())
            if idx < len(items):
                f.seek(0)
                f.write(str(idx + 1))
                f.truncate()
        if idx >= len(items):
            break
        yield idx, items[idx]
    dist.barrier()


class FailureLog:
    """Keep going after a failed job, but finish with exit code 1.

    The launchers in the repository root read that exit code, so a crash in
    the middle of a run (OOM, a bug) shows up in their summary.
    """

    def __init__(self, rank=0):
        self.rank = rank
        self.labels = []

    def run(self, label, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
            return True
        except Exception as e:
            print(f"[GPU {self.rank}] Error on {label}: {e}")
            traceback.print_exc()
            self.labels.append(label)
            return False

    def finish(self):
        """Tear down DDP and exit 1 if a job failed on any rank.

        Under torchrun every rank waits here for the others and all exit with
        the same code: one rank exiting non-zero while others still train
        would make torchrun kill them and lose their unsaved results.
        """
        failed = bool(self.labels)
        if dist.is_initialized():
            flag = torch.tensor([int(failed)])
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            failed = bool(flag.item())
            dist.destroy_process_group()
        if self.labels:
            print(f"\n[GPU {self.rank}] {len(self.labels)} job(s) failed:")
            for label in self.labels:
                print(f"   {label}")
        if failed:
            sys.exit(1)
