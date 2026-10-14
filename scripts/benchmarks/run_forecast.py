"""
Dataset-parallel Forecasting Script for Time Series SSL Baselines

One continuous training run per (dataset, seed). At each checkpoint epoch the
series is encoded causally and a Ridge head is fit per horizon; one row per
horizon goes to the result CSV. Under torchrun each GPU takes the next
unclaimed dataset.

Usage:
    python scripts/benchmarks/run_forecast.py --method ts2vec --all_datasets --iterations 3
    torchrun --nproc_per_node=2 scripts/benchmarks/run_forecast.py --method ts2vec --all_datasets --iterations 3

Supported methods: ts2vec, softclt, timesurl, trep, autotcl, infots, cost, simmtm, ts_jepa, fei,
                   cf_jepa (ours)
"""

import argparse
import os
import sys
import warnings

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import numpy as np

from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baselines import get_trainer
from utils.data_loaders import load_forecast_csv
from utils.result_status import FCST_CSV, default_seeds, dim_overrides, dims_for, fcst_done
from utils.runner import (
    FailureLog, append_csv_rows, auto_max_train_length, banner, dual_encoder_ready,
    get_device, iter_work, main_print, parse_epochs, set_seed, setup_ddp,
    train_with_checkpoints,
)

# Paper scope (electricity excluded)
FORECAST_DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'AirQuality', 'SML', 'Traffic']
CSV_FIELDNAMES = ['method', 'dataset', 'seed', 'epoch', 'hidden_dims', 'repr_dims',
                  'univar', 'pred_len', 'mse', 'mae', 'alpha']


# =============================================================================
# Causal Encoding
# =============================================================================

def encode_causal(trainer, data, window=200, min_segment_len=210, use_target=True):
    """Causal encoding: for each timestep t, encode segment [t-window+1 : t+1]
    and take the last timestep's representation.

    Uses trainer.encode(raw=True) for (1, T_seg, D) and takes the last timestep;
    pooled-only methods (SimMTM) return (1, D), used as is.

    Segments shorter than min_segment_len are left-padded with zeros to avoid
    FFT dimension errors (e.g., CoST's BandedFourierLayer expects length >= 201).

    `use_target` picks the EMA target encoder for trainers that have one (the
    CF-JEPA family). Every other baseline, including TS-JEPA and FEI (which keep
    their EMA target internal), has a single encoder and ignores it.

    Returns: np.ndarray, shape (n_instances, T, D)
    """
    n_instances, T, C = data.shape
    enc_kwargs = {'use_target': use_target} if dual_encoder_ready(trainer) else {}
    all_reprs = []

    for inst_idx in range(n_instances):
        inst_data = data[inst_idx:inst_idx+1]
        reprs = []

        for t_i in range(T):
            start = max(0, t_i - window + 1)
            segment = inst_data[:, start:t_i+1, :]

            seg_len = segment.shape[1]
            if seg_len < min_segment_len:
                pad = np.zeros((1, min_segment_len - seg_len, C))
                segment = np.concatenate([pad, segment], axis=1)

            z = trainer.encode(segment, raw=True, **enc_kwargs)
            reprs.append(z[0, -1] if z.ndim == 3 else z[0])  # (D,)

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
# Train + evaluate at checkpoints
# =============================================================================

def run_forecast_checkpoints(method, dataset, checkpoint_epochs, device='cuda', univar=False,
                             window=200, seed=42, eval_csv=FCST_CSV, **method_kwargs):
    """Train once; at each checkpoint encode causally and fit a Ridge head per horizon."""
    set_seed(seed)
    hidden_dims, repr_dims = dims_for(method, method_kwargs.get('hidden_dims'),
                                      method_kwargs.get('output_dims'))
    banner(f"Dataset: {dataset} ({'univar' if univar else 'multivar'})", f"Method: {method}",
           f"Checkpoints: {checkpoint_epochs}", f"Seed: {seed}", f"Device: {device}",
           f"Hidden dims: {hidden_dims}", f"Repr dims: {repr_dims}")

    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = \
        load_forecast_csv(dataset, univar=univar)
    main_print(f"Data shape: {data.shape}")
    train_data = data[:, train_slice]
    valid_data = data[:, valid_slice]
    test_data = data[:, test_slice]

    auto_max_train_length(method_kwargs, train_data.shape[1])
    trainer = get_trainer(method=method, input_dims=data.shape[2], device=device,
                          dataset=dataset, **method_kwargs)

    # Main rows use the EMA target encoder where one exists (asymmetric dual
    # deployment). Dual-encoder eval: cf_jepa also reports its online encoder
    # as 'cf_jepa_online'.
    online_label = 'cf_jepa_online' if method == 'cf_jepa' else None
    eval_results = []

    def encode_and_score(tr, ep, row_method, use_target):
        train_repr = encode_causal(tr, train_data, window=window, use_target=use_target)
        valid_repr = encode_causal(tr, valid_data, window=window, use_target=use_target)
        test_repr = encode_causal(tr, test_data, window=window, use_target=use_target)
        tag = '' if row_method == method else f'[{row_method}] '
        for pred_len in pred_lens:
            metrics = fit_ridge(train_repr, train_data, valid_repr, valid_data,
                                test_repr, test_data, pred_len, n_covariate_cols)
            eval_results.append({
                'method': row_method,
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
            main_print(f"    {tag}H={pred_len}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

    def evaluate(tr, ep):
        main_print(f"\n  [Checkpoint] Encoding (causal, window={window}) at epoch {ep}...")
        encode_and_score(tr, ep, method, use_target=True)
        if online_label and dual_encoder_ready(tr):
            encode_and_score(tr, ep, online_label, use_target=False)

    mode = 'univar' if univar else 'multivar'
    loss_csv = f'results/{method}/forecasting/{dataset}/loss_logs/seed{seed}_{mode}_loss.csv'
    train_time = train_with_checkpoints(trainer, train_data, checkpoint_epochs, evaluate, loss_csv)
    append_csv_rows(eval_csv, CSV_FIELDNAMES, eval_results)
    main_print(f"  Eval results: {eval_csv}")
    main_print(f"  Total train time: {train_time:.1f}s")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dataset-parallel Forecasting for Time Series SSL')
    parser.add_argument('--method', type=str, default='ts2vec',
                        help='ts2vec, softclt, timesurl, trep, autotcl, infots, cost, simmtm, '
                             'ts_jepa, fei, cf_jepa')
    parser.add_argument('--dataset', type=str, default=None, action='append')
    parser.add_argument('--all_datasets', action='store_true',
                        help='All paper datasets (FORECAST_DATASETS)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of seeds (42, 43, ...)')
    parser.add_argument('--univar', action='store_true')
    parser.add_argument('--window', type=int, default=200)
    parser.add_argument('--checkpoint_epochs', type=str, default='50,100,200,400',
                        help='Comma-separated checkpoint epochs (default: 50,100,200,400)')
    parser.add_argument('--hidden_dims', type=int, default=None,
                        help='Override hidden dims (default: per-method)')
    parser.add_argument('--output_dims', type=int, default=None,
                        help='Override output/repr dims (default: per-method)')
    parser.add_argument('--eval_csv', type=str, default=FCST_CSV,
                        help=f'Output CSV path (default: {FCST_CSV}).')
    args = parser.parse_args()

    checkpoint_epochs = parse_epochs(args.checkpoint_epochs)
    method_kwargs = dim_overrides(args.hidden_dims, args.output_dims)
    hidden_dims, repr_dims = dims_for(args.method, args.hidden_dims, args.output_dims)

    rank, world_size, local_rank = setup_ddp()
    device = get_device(local_rank)

    if args.all_datasets:
        all_datasets = list(FORECAST_DATASETS)
    else:
        all_datasets = args.dataset or ['ETTh1']
    if 'electricity' in all_datasets:
        main_print("[WARN] 'electricity' is out of paper scope - excluding from run")
        all_datasets = [d for d in all_datasets if d != 'electricity']

    main_print(f"Method: {args.method}, Datasets: {len(all_datasets)} total, "
               f"Checkpoints: {checkpoint_epochs}, World size: {world_size}")

    failures = FailureLog(rank)
    for iteration, seed in enumerate(default_seeds(args.iterations), start=1):
        main_print(f"\n>>> ITERATION {iteration}/{args.iterations} | Seed: {seed}")

        for idx, dataset in iter_work(all_datasets, rank, world_size, f'fcst_{args.method}'):
            if fcst_done(args.eval_csv, args.method, dataset, seed, max(checkpoint_epochs),
                         args.univar, hidden_dims, repr_dims):
                print(f"  [Skip] {dataset} (seed {seed} already done)")
                continue
            ok = failures.run(
                f'{dataset} (seed {seed})', run_forecast_checkpoints,
                method=args.method, dataset=dataset, checkpoint_epochs=checkpoint_epochs,
                device=device, univar=args.univar, window=args.window, seed=seed,
                eval_csv=args.eval_csv, **method_kwargs,
            )
            if ok:
                print(f"\n[GPU {rank}] [{idx + 1}/{len(all_datasets)}] Done: {dataset} (seed {seed})\n")

        main_print(f"\n{'='*70}")
        main_print(f"{args.method.upper()} FORECAST ITERATION {iteration} COMPLETE")
        main_print(f"{'='*70}")

    failures.finish()


if __name__ == '__main__':
    main()
