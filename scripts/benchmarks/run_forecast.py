"""
Dataset-parallel Forecasting Script for Time Series SSL Baselines

Usage:
    python scripts/benchmarks/run_forecast.py --method ts2vec --all_ett --epochs 200 --iterations 3
    torchrun --nproc_per_node=2 scripts/benchmarks/run_forecast.py --method ts2vec --all_ett --epochs 200 --iterations 3

Supported methods: ts2vec, softclt, timesurl, trep, autotcl, infots, cost, simmtm, ts_jepa, fei, cf_jepa
"""

import argparse
import os
import sys
import csv
import time
import warnings

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import numpy as np

import torch
import torch.distributed as dist
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baselines import get_trainer, get_method_dims
from utils.data_loaders import load_forecast_csv


# =============================================================================
# DDP Helpers
# =============================================================================

def setup_ddp():
    if 'RANK' in os.environ:
        backend = 'gloo'
        from datetime import timedelta
        dist.init_process_group(backend=backend, timeout=timedelta(hours=24))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# =============================================================================
# Causal Encoding
# =============================================================================

def encode_causal(trainer, data, batch_size=256, window=200, min_segment_len=210,
                  use_target=True):
    """Causal encoding: for each timestep t, encode segment [t-window+1 : t+1]
    and take the last timestep's representation.

    Uses trainer.encode(raw=True) for (1, T_seg, D) → take last timestep,
    falling back to trainer.encode() → (1, D) for methods without raw support.

    Segments shorter than min_segment_len are left-padded with zeros to avoid
    FFT dimension errors (e.g., CoST's BandedFourierLayer expects length >= 201).

    `use_target=True` (default) selects the EMA target encoder, which only the
    CF-JEPA family exposes (`target_encoder` attribute). Every other baseline —
    including TS-JEPA and FEI, which have EMA targets upstream but do not surface
    them through this wrapper — ignores the flag and encodes with its single
    encoder. For CF-JEPA dual-encoder cross-eval pass `use_target=False` to force
    the online encoder.

    Returns: np.ndarray, shape (n_instances, T, D)
    """
    n_instances, T, C = data.shape
    all_reprs = []
    has_target = hasattr(trainer, 'target_encoder') and getattr(trainer, 'target_encoder') is not None

    for inst_idx in range(n_instances):
        inst_data = data[inst_idx:inst_idx+1]
        reprs = []

        for t_i in range(T):
            start = max(0, t_i - window + 1)
            segment = inst_data[:, start:t_i+1, :]

            # Left-pad short segments to avoid encoder errors (e.g., CoST FFT)
            seg_len = segment.shape[1]
            if seg_len < min_segment_len:
                pad_len = min_segment_len - seg_len
                pad = np.zeros((1, pad_len, C))
                segment = np.concatenate([pad, segment], axis=1)

            # Try raw=True first: (1, T_seg, D) → take last timestep
            if has_target:
                z = trainer.encode(segment, raw=True, use_target=use_target)
            else:
                z = trainer.encode(segment, raw=True)
            if z.ndim == 3:
                reprs.append(z[0, -1])  # (D,)
            else:
                # Already pooled (e.g., SimMTM) → (1, D) → take first
                reprs.append(z[0])  # (D,)

        all_reprs.append(np.array(reprs))

    return np.array(all_reprs)


# =============================================================================
# Forecasting Evaluation
# =============================================================================

def generate_pred_samples(features, data, pred_len, drop=0):
    n_instances, T, D = features.shape
    X_list, Y_list = [], []
    for i in range(n_instances):
        for t in range(drop, T - pred_len):
            X_list.append(features[i, t])
            Y_list.append(data[i, t+1:t+1+pred_len].flatten())
    return np.array(X_list), np.array(Y_list)


def _count_pred_samples(features, pred_len, drop=0):
    """Count total samples without allocating arrays."""
    n_instances, T, D = features.shape
    count = 0
    for i in range(n_instances):
        count += max(0, T - pred_len - drop)
    return count


def _generate_pred_samples_subsampled(features, data, pred_len, max_samples, drop=0, random_state=0):
    """Generate prediction samples with pre-subsampling to avoid OOM."""
    total = _count_pred_samples(features, pred_len, drop)
    if total == 0:
        return np.array([]), np.array([])
    if total <= max_samples:
        return generate_pred_samples(features, data, pred_len, drop)

    # Pre-select indices to keep
    rng = np.random.RandomState(random_state)
    keep = set(rng.choice(total, size=max_samples, replace=False).tolist())

    n_instances, T, D = features.shape
    X_list, Y_list = [], []
    idx = 0
    for i in range(n_instances):
        for t in range(drop, T - pred_len):
            if idx in keep:
                X_list.append(features[i, t])
                Y_list.append(data[i, t+1:t+1+pred_len].flatten())
            idx += 1
    return np.array(X_list), np.array(Y_list)


def fit_ridge(train_features, train_data, valid_features, valid_data,
              test_features, test_data, pred_len, n_covariate_cols=0,
              padding=200, MAX_SAMPLES=100000):
    train_target = train_data[:, :, n_covariate_cols:]
    valid_target = valid_data[:, :, n_covariate_cols:]
    test_target = test_data[:, :, n_covariate_cols:]

    train_X, train_Y = _generate_pred_samples_subsampled(
        train_features, train_target, pred_len, MAX_SAMPLES, drop=padding)
    valid_X, valid_Y = _generate_pred_samples_subsampled(
        valid_features, valid_target, pred_len, MAX_SAMPLES)
    test_X, test_Y = generate_pred_samples(test_features, test_target, pred_len)

    if len(train_X) == 0 or len(test_X) == 0:
        return {'mse': float('inf'), 'mae': float('inf')}

    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha).fit(train_X, train_Y)
        valid_pred = ridge.predict(valid_X)
        score = np.sqrt(((valid_pred - valid_Y) ** 2).mean()) + np.abs(valid_pred - valid_Y).mean()
        valid_results.append(score)
    best_alpha = alphas[np.argmin(valid_results)]

    ridge = Ridge(alpha=best_alpha)
    ridge.fit(train_X, train_Y)
    test_pred = ridge.predict(test_X)

    mse = np.mean((test_pred - test_Y) ** 2)
    mae = np.mean(np.abs(test_pred - test_Y))

    return {'mse': mse, 'mae': mae, 'alpha': best_alpha}


# =============================================================================
# Main
# =============================================================================

FORECAST_DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'AirQuality', 'SML', 'Traffic']


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_forecast_checkpoint(method, dataset, max_epochs=400, checkpoint_epochs=None,
                            device='cuda', univar=False, window=200,
                            local_rank=-1, seed=42, hidden_dims=None, repr_dims=None,
                            eval_csv=None, **method_kwargs):
    """Train and evaluate forecasting at each checkpoint epoch.

    At each checkpoint: causal encode → Ridge regression → append to CSV.
    No npz files saved.
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = [50, 100, 200, 400]

    set_seed(seed)
    rank0 = is_main_process()

    if rank0:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({'univar' if univar else 'multivar'})")
        print(f"Method: {method}")
        print(f"Max Epochs: {max_epochs}")
        print(f"Checkpoints: {checkpoint_epochs}")
        print(f"Seed: {seed}")
        print(f"Device: {device}")
        print(f"Hidden dims: {hidden_dims}")
        print(f"Repr dims: {repr_dims}")
        print(f"{'='*60}")

    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = \
        load_forecast_csv(dataset, univar=univar)

    if rank0:
        print(f"Data shape: {data.shape}")

    train_data = data[:, train_slice]
    valid_data = data[:, valid_slice]
    test_data = data[:, test_slice]
    input_dims = data.shape[2]

    T = train_data.shape[1]
    if 'max_train_length' not in method_kwargs and T > 2048:
        method_kwargs['max_train_length'] = 2048
        if rank0:
            print(f"  Auto max_train_length=2048 (T={T})")

    trainer = get_trainer(method=method, input_dims=input_dims, device=device,
                          dataset=dataset, **method_kwargs)

    # Results CSV
    if eval_csv is None:
        eval_csv = 'results/fcst_results.csv'
    os.makedirs(os.path.dirname(eval_csv) or '.', exist_ok=True)

    loss_log = []
    eval_results = []
    t_start = time.time()

    # Single continuous fit() call; callback triggers encode+eval at checkpoint epochs.
    # See run_classification.py for rationale (avoids optimizer/scheduler reset).
    sorted_checkpoints = sorted(ep for ep in checkpoint_epochs if ep <= max_epochs)
    if not sorted_checkpoints:
        return 0.0
    max_checkpoint = sorted_checkpoints[-1]
    checkpoint_set = set(sorted_checkpoints)

    def _epoch_callback(tr, ep, loss):
        loss_log.append((ep, loss))
        if ep not in checkpoint_set:
            return

        if rank0:
            print(f"\n  [Checkpoint] Encoding (causal, window={window}) at epoch {ep}...")
        # Default forecast eval uses target encoder (asymmetric dual deployment).
        train_repr = encode_causal(tr, train_data, window=window)
        valid_repr = encode_causal(tr, valid_data, window=window)
        test_repr = encode_causal(tr, test_data, window=window)

        for pred_len in pred_lens:
            metrics = fit_ridge(train_repr, train_data, valid_repr, valid_data,
                                test_repr, test_data, pred_len, n_covariate_cols)
            eval_results.append({
                'method': method,
                'dataset': dataset,
                'seed': seed,
                'epoch': ep,
                'hidden_dims': hidden_dims,
                'repr_dims': repr_dims,
                'univar': univar,
                'pred_len': pred_len,
                'mse': round(metrics['mse'], 4),
                'mae': round(metrics['mae'], 4),
                'alpha': metrics.get('alpha', ''),
            })
            if rank0:
                print(f"    H={pred_len}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

        del train_repr, valid_repr, test_repr

        # Dual-encoder eval: for cf_jepa, also evaluate with the online
        # encoder. Saved as method='cf_jepa_online'. Same training, just extra
        # causal encoding pass.
        if (method == 'cf_jepa'
                and hasattr(tr, 'target_encoder')
                and tr.target_encoder is not None):
            if rank0:
                print(f"  [Dual eval] Re-encoding with online encoder...")
            train_repr_o = encode_causal(tr, train_data, window=window, use_target=False)
            valid_repr_o = encode_causal(tr, valid_data, window=window, use_target=False)
            test_repr_o = encode_causal(tr, test_data, window=window, use_target=False)

            for pred_len in pred_lens:
                metrics_o = fit_ridge(
                    train_repr_o, train_data, valid_repr_o, valid_data,
                    test_repr_o, test_data, pred_len, n_covariate_cols)
                eval_results.append({
                    'method': 'cf_jepa_online',
                    'dataset': dataset,
                    'seed': seed,
                    'epoch': ep,
                    'hidden_dims': hidden_dims,
                    'repr_dims': repr_dims,
                    'univar': univar,
                    'pred_len': pred_len,
                    'mse': round(metrics_o['mse'], 4),
                    'mae': round(metrics_o['mae'], 4),
                    'alpha': metrics_o.get('alpha', ''),
                })
                if rank0:
                    print(f"    [online] H={pred_len}: MSE={metrics_o['mse']:.4f}, MAE={metrics_o['mae']:.4f}")
            del train_repr_o, valid_repr_o, test_repr_o

        import gc; gc.collect()

    if rank0:
        print(f"\n  Training 1 -> {max_checkpoint} (continuous)...")
    trainer.fit(train_data, epochs=max_checkpoint, verbose=False,
                callback=_epoch_callback)

    train_time = time.time() - t_start

    # Append eval results to CSV
    if eval_results:
        fieldnames = ['method', 'dataset', 'seed', 'epoch', 'hidden_dims', 'repr_dims',
                      'univar', 'pred_len', 'mse', 'mae', 'alpha']
        file_exists = os.path.exists(eval_csv)
        with open(eval_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(eval_results)

    # Save loss log
    loss_dir = f'results/{method}/forecasting/{dataset}/loss_logs'
    os.makedirs(loss_dir, exist_ok=True)
    mode_str = 'univar' if univar else 'multivar'
    loss_csv = f'{loss_dir}/seed{seed}_{mode_str}_loss.csv'
    with open(loss_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss'])
        writer.writeheader()
        for ep, loss in loss_log:
            writer.writerow({'epoch': ep, 'loss': round(loss, 6)})

    if rank0:
        print(f"\n  Loss log: {loss_csv}")
        print(f"  Eval results: {eval_csv}")
        print(f"  Total train time: {train_time:.1f}s")

    return train_time


def _is_forecast_done(method, dataset, seed, max_epoch, univar=False,
                      hidden_dims=None, repr_dims=None,
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


def main():
    parser = argparse.ArgumentParser(description='Dataset-parallel Forecasting for Time Series SSL')
    parser.add_argument('--method', type=str, default='ts2vec',
                        help='ts2vec, softclt, timesurl, trep, autotcl, infots, cost, simmtm, ts_jepa, fei')
    parser.add_argument('--dataset', type=str, default=None, action='append')
    parser.add_argument('--all_ett', action='store_true')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--univar', action='store_true')
    parser.add_argument('--window', type=int, default=200)
    parser.add_argument('--checkpoint_epochs', type=str, default='50,100,200,400',
                        help='Comma-separated checkpoint epochs (default: 50,100,200,400)')
    parser.add_argument('--hidden_dims', type=int, default=None,
                        help='Override hidden dims (default: per-method)')
    parser.add_argument('--eval_csv', type=str, default='results/fcst_results.csv',
                        help='Output CSV path (default: results/fcst_results.csv).')
    parser.add_argument('--output_dims', type=int, default=None,
                        help='Override output/repr dims (default: per-method)')
    args = parser.parse_args()

    checkpoint_epochs = [int(x) for x in args.checkpoint_epochs.split(',')]
    max_epoch = max(checkpoint_epochs)

    # Build method_kwargs from CLI overrides
    method_kwargs = {}
    if args.hidden_dims is not None:
        method_kwargs['hidden_dims'] = args.hidden_dims
    if args.output_dims is not None:
        method_kwargs['output_dims'] = args.output_dims

    # NOTE: forecast_mode flag removed 2026-05-29 — AutoTCL unified to TS2Vec
    # backbone + class defaults across cls/fcst/ano (see baselines/autotcl/
    # trainer.py docstring). Per-task backbone branching gave AutoTCL a unique
    # advantage; single-backbone fairness aligned with the other TS2Vec-family
    # baselines (TS2Vec, SoftCLT, TimesURL, T-Rep, InfoTS).

    # Get hidden_dims/repr_dims for CSV recording
    default_h, default_r = get_method_dims(args.method)
    actual_hidden = args.hidden_dims if args.hidden_dims is not None else default_h
    actual_repr = args.output_dims if args.output_dims is not None else default_r

    rank, world_size, local_rank = setup_ddp()
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    if args.all_ett:
        all_datasets = FORECAST_DATASETS
    elif args.dataset:
        all_datasets = args.dataset
    else:
        all_datasets = ['ETTh1']

    # Paper scope: electricity is out of scope — drop unconditionally.
    if 'electricity' in all_datasets:
        if rank == 0:
            print("[WARN] 'electricity' is out of paper scope — excluding from run")
        all_datasets = [d for d in all_datasets if d != 'electricity']

    # Dynamic queue for multi-GPU
    use_queue = world_size > 1
    if use_queue:
        import tempfile, fcntl
        counter_file = os.path.join(tempfile.gettempdir(),
            f'ssl_fcst_{args.method}_{os.environ.get("MASTER_PORT", "29500")}.counter')
        print(f"[GPU {rank}] Dynamic queue mode, {len(all_datasets)} datasets")
    else:
        print(f"Method: {args.method}, Datasets: {len(all_datasets)} total, Checkpoints: {checkpoint_epochs}")

    def get_next_dataset_idx():
        with open(counter_file, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            idx = int(f.read().strip())
            if idx < len(all_datasets):
                f.seek(0)
                f.write(str(idx + 1))
                f.truncate()
            fcntl.flock(f, fcntl.LOCK_UN)
        return idx if idx < len(all_datasets) else None

    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    for iteration in range(1, args.iterations + 1):
        seed = seeds[iteration - 1] if iteration <= len(seeds) else 42 + iteration - 1

        if rank == 0:
            print(f"\n>>> ITERATION {iteration}/{args.iterations} | Seed: {seed}")

        if use_queue:
            if rank == 0:
                with open(counter_file, 'w') as f:
                    f.write('0')
            dist.barrier()

            while True:
                idx = get_next_dataset_idx()
                if idx is None:
                    break
                dataset = all_datasets[idx]
                if _is_forecast_done(args.method, dataset, seed, max_epoch,
                                     univar=args.univar, hidden_dims=actual_hidden,
                                     repr_dims=actual_repr, eval_csv=args.eval_csv):
                    continue
                try:
                    run_forecast_checkpoint(
                        method=args.method, dataset=dataset,
                        max_epochs=max_epoch, checkpoint_epochs=checkpoint_epochs,
                        device=device, univar=args.univar, window=args.window,
                        local_rank=local_rank, seed=seed,
                        hidden_dims=actual_hidden, repr_dims=actual_repr,
                        eval_csv=args.eval_csv,
                        **method_kwargs,
                    )
                    print(f"\n[GPU {rank}] [{idx + 1}/{len(all_datasets)}] Done: {dataset} (seed {seed})\n")
                except Exception as e:
                    print(f"[GPU {rank}] Error on {dataset}: {e}")
                    import traceback
                    traceback.print_exc()

            dist.barrier()
        else:
            for dataset in all_datasets:
                if _is_forecast_done(args.method, dataset, seed, max_epoch,
                                     univar=args.univar, hidden_dims=actual_hidden,
                                     repr_dims=actual_repr, eval_csv=args.eval_csv):
                    print(f"  [Skip] {dataset} (seed {seed} already done)")
                    continue
                try:
                    run_forecast_checkpoint(
                        method=args.method, dataset=dataset,
                        max_epochs=max_epoch, checkpoint_epochs=checkpoint_epochs,
                        device=device, univar=args.univar, window=args.window,
                        local_rank=local_rank, seed=seed,
                        hidden_dims=actual_hidden, repr_dims=actual_repr,
                        eval_csv=args.eval_csv,
                        **method_kwargs,
                    )
                except Exception as e:
                    print(f"Error on {dataset}: {e}")
                    import traceback
                    traceback.print_exc()

        if rank == 0:
            print(f"\n{'='*70}")
            print(f"{args.method.upper()} FORECAST ITERATION {iteration} COMPLETE")
            print(f"{'='*70}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
