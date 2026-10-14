"""
Dataset-parallel Anomaly Detection Script for Time Series SSL Baselines

Approach:
    1. Train SSL encoder on normal (training) data
    2. Encode train+test series → per-timestep representations
    3. Score anomalies using representation-based methods (knn, gaussian)
    4. Threshold on training scores (mean + 4*std)
    5. Evaluate with delay-adjusted F1/Precision/Recall
       (multivariate: same protocol plus comparison variants, see eval_mv_protocols)

One dataset per call; under torchrun the seeds are split across ranks.

Usage:
    python scripts/benchmarks/run_anomaly.py --method ts2vec --dataset kpi --iterations 3
    torchrun --nproc_per_node=2 scripts/benchmarks/run_anomaly.py --method ts2vec --dataset kpi
"""

import argparse
import os
import sys
import warnings

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baselines import get_trainer
from utils.anomaly_eval import (
    SCORERS, eval_ad_result, eval_mv_protocols, suppress_within_delay, threshold_train4sigma,
)
from utils.data_loaders import (
    ANOMALY_MV_DATASETS, gen_ano_mv_train_data, gen_ano_train_data, load_anomaly,
    load_anomaly_mv,
)
from utils.result_status import (
    ANO_CSV, ANO_MV_CSV, ano_done, default_seeds, dim_overrides, dims_for,
)
from utils.runner import (
    FailureLog, append_csv_rows, auto_max_train_length, banner, dual_encoder_ready,
    get_device, main_print, parse_epochs, set_seed, setup_ddp, train_with_checkpoints,
)

# Default anomaly ratios per dataset (Anomaly-Transformer / DCdetector convention).
MV_ANOMALY_RATIO = {
    'SMD': 0.005,
    'MSL': 0.01,
    'SMAP': 0.01,
    'PSM': 0.01,
}

# Delay tolerance for the multivariate benchmarks. All four are sampled at
# one-minute resolution, so they take KPI's value (Yahoo is hourly, delay 3).
# MV_EXTRA_DELAYS are recorded alongside for a delay-sensitivity check only.
MV_DELAY = 7
MV_EXTRA_DELAYS = (3,)

UV_FIELDNAMES = ['method', 'dataset', 'seed', 'epoch', 'hidden_dims', 'repr_dims',
                 'scoring', 'f1', 'precision', 'recall']
MV_FIELDNAMES = ['method', 'dataset', 'seed', 'epoch', 'hidden_dims', 'repr_dims',
                 'scoring', 'protocol', 'delay', 'f1', 'precision', 'recall']


# =============================================================================
# Encoding
# =============================================================================

def encode_sequence(trainer, x, max_chunk=5000, overlap=200, min_len=210,
                    use_target=False):
    """Encode one series, (T,) or (T, C), into per-timestep representations (T, D).

    Short inputs are left-padded to `min_len`; long ones are encoded in
    overlapping chunks and stitched. Outputs shorter than the input (TS-JEPA
    patches, FEI strides) are linearly interpolated back to T. Pooled-only
    methods (e.g. SimMTM) return (D,). `use_target=True` is only valid for
    trainers with an EMA target encoder (CF-JEPA family).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    T, C = x.shape
    enc_kwargs = {'use_target': True} if use_target else {}

    def _encode_chunk(chunk):
        L = chunk.shape[0]
        data = chunk.reshape(1, L, C).astype(np.float32)
        if L < min_len:
            pad = np.zeros((1, min_len - L, C), dtype=np.float32)
            data = np.concatenate([pad, data], axis=1)
        z = trainer.encode(data, raw=True, **enc_kwargs)
        if z.ndim != 3:
            return z.squeeze()  # (D,) — pooled-only method
        z = z[0]
        if L < min_len:
            z = z[-L:]  # strip padding
        if z.shape[0] != L:
            z = _interpolate_repr(z, L)
        return z

    if T <= max_chunk:
        return _encode_chunk(x)

    chunks = []
    for start in range(0, T, max_chunk - overlap):
        end = min(start + max_chunk, T)
        z = _encode_chunk(x[start:end])
        if z.ndim == 1:
            # Pooled method — can't do per-timestep with chunking
            return trainer.encode(x.reshape(1, T, C).astype(np.float32), raw=False,
                                  **enc_kwargs).squeeze()
        chunks.append(z if start == 0 else z[overlap:])  # skip the overlap region
        if end >= T:
            break

    return np.concatenate(chunks, axis=0)[:T]  # ensure exact length


def _interpolate_repr(z, target_len):
    """Linearly interpolate (T', D) representations to (target_len, D)."""
    from scipy.interpolate import interp1d
    src_len = z.shape[0]
    if src_len == target_len:
        return z
    f = interp1d(np.linspace(0, 1, src_len), z, axis=0, kind='linear')
    return f(np.linspace(0, 1, target_len))


def _row_labels(method, trainer):
    """(online label, target label or None) for the CSV rows of one checkpoint.

    The paper reports CF-JEPA's EMA target encoder for anomaly detection, so
    its target rows carry the plain method name and the online rows get
    '<method>_online' (as in the forecasting runner). Single-encoder methods
    have one row under their own name.
    """
    if method.startswith('cf_jepa') and dual_encoder_ready(trainer):
        return f'{method}_online', method
    return method, None


def _evaluate_both_encoders(tr, ep, method, encode_and_score):
    """Online rows, then target rows. The order is fixed: k-NN subsampling
    draws from the global numpy RNG, so swapping it changes the scores."""
    online_label, target_label = _row_labels(method, tr)
    if not encode_and_score(tr, ep, online_label, use_target=False):
        main_print(f"    Skipped: {method} has no per-timestep representations")
        return
    if target_label:
        encode_and_score(tr, ep, target_label, use_target=True)


# =============================================================================
# Train + evaluate at checkpoints
# =============================================================================

def run_anomaly_checkpoints(method, dataset, checkpoint_epochs, device='cuda', seed=42,
                            scoring='knn', eval_csv=ANO_CSV, **method_kwargs):
    """Univariate (KPI / Yahoo): train once on all training series; at each
    checkpoint score every series and write one delay-adjusted F1 row."""
    set_seed(seed)
    hidden_dims, repr_dims = dims_for(method, method_kwargs.get('hidden_dims'),
                                      method_kwargs.get('output_dims'))
    banner(f"Anomaly Detection: {dataset}", f"Method: {method} | Scoring: {scoring}",
           f"Checkpoints: {checkpoint_epochs}", f"Seed: {seed} | Device: {device}",
           f"Hidden dims: {hidden_dims} | Repr dims: {repr_dims}")

    all_train_data, all_train_labels, all_train_timestamps, \
        all_test_data, all_test_labels, all_test_timestamps, delay = load_anomaly(dataset)
    series_keys = sorted(all_train_data.keys())
    main_print(f"  Series count: {len(series_keys)}")
    main_print(f"  Delay: {delay}")
    main_print(f"  Total train points: {sum(len(all_train_data[k]) for k in series_keys):,}")
    main_print(f"  Total test points: {sum(len(all_test_data[k]) for k in series_keys):,}")

    pretrain_data = gen_ano_train_data(all_train_data)  # (n_series, max_T, 1)
    main_print(f"  Pre-train data shape: {pretrain_data.shape}")

    auto_max_train_length(method_kwargs, pretrain_data.shape[1])
    trainer = get_trainer(method=method, input_dims=1, device=device,
                          dataset=dataset, **method_kwargs)
    scorer = SCORERS[scoring]
    eval_results = []

    def encode_and_score(tr, ep, row_method, use_target):
        preds, labels, timestamps = [], [], []
        for ki, k in enumerate(series_keys):
            train_series = all_train_data[k]
            full_repr = encode_sequence(tr, np.concatenate([train_series, all_test_data[k]]),
                                        use_target=use_target)
            if full_repr.ndim == 1:
                return False
            train_scores, test_scores = scorer(full_repr[:len(train_series)],
                                               full_repr[len(train_series):])
            pred = (test_scores > threshold_train4sigma(train_scores)).astype(np.int64)
            preds.append(suppress_within_delay(pred, delay))
            labels.append(all_test_labels[k])
            timestamps.append(all_test_timestamps[k])
            if (ki + 1) % 50 == 0:
                main_print(f"    Processed {ki+1}/{len(series_keys)} series")

        m = eval_ad_result(preds, labels, timestamps, delay)
        eval_results.append({
            'method': row_method,
            'dataset': dataset,
            'seed': seed,
            'epoch': ep,
            'hidden_dims': hidden_dims,
            'repr_dims': repr_dims,
            'scoring': scoring,
            'f1': round(m['f1'], 4),
            'precision': round(m['precision'], 4),
            'recall': round(m['recall'], 4),
        })
        tag = '' if row_method == method else f'[{row_method}] '
        main_print(f"    {tag}F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")
        return True

    def evaluate(tr, ep):
        main_print(f"\n  [Checkpoint] Encoding and scoring at epoch {ep}...")
        _evaluate_both_encoders(tr, ep, method, encode_and_score)

    loss_csv = f'results/{method}/anomaly/{dataset}/loss_logs/seed{seed}_loss.csv'
    train_time = train_with_checkpoints(trainer, pretrain_data, checkpoint_epochs, evaluate, loss_csv)
    append_csv_rows(eval_csv, UV_FIELDNAMES, eval_results)
    main_print(f"  Eval results: {eval_csv}")
    main_print(f"  Total time: {train_time:.1f}s")


def run_anomaly_mv_checkpoints(method, dataset, checkpoint_epochs, device='cuda', seed=42,
                               scorings=('knn', 'gaussian'), eval_csv=ANO_MV_CSV,
                               window_size=100, anomaly_ratio=None, delay=MV_DELAY,
                               extra_delays=MV_EXTRA_DELAYS, **method_kwargs):
    """Multivariate (SMD / MSL / SMAP / PSM): train once on non-overlapping
    `window_size` windows of the training series; at each checkpoint encode
    train+test once and write one row per (scoring, protocol) — see
    `eval_mv_protocols`. `anomaly_ratio` (default MV_ANOMALY_RATIO[dataset])
    only affects the *_ratio protocols.
    """
    if anomaly_ratio is None:
        anomaly_ratio = MV_ANOMALY_RATIO.get(dataset, 0.01)
    set_seed(seed)
    hidden_dims, repr_dims = dims_for(method, method_kwargs.get('hidden_dims'),
                                      method_kwargs.get('output_dims'))
    banner(f"Anomaly Detection (multivariate): {dataset}",
           f"Method: {method} | Scoring: {list(scorings)}",
           f"Checkpoints: {checkpoint_epochs}", f"Seed: {seed} | Device: {device}",
           f"Hidden dims: {hidden_dims} | Repr dims: {repr_dims}",
           f"Window: {window_size} | Anomaly ratio: {anomaly_ratio} | Delay: {delay}")

    train_x, test_x, test_label = load_anomaly_mv(dataset)
    T_train, D = train_x.shape
    T_test = test_x.shape[0]
    main_print(f"  Train: ({T_train}, {D})  Test: ({T_test}, {D})  "
               f"anomaly={test_label.mean():.4f}")

    pretrain_windows = gen_ano_mv_train_data(train_x, window_size=window_size,
                                             stride=window_size)
    main_print(f"  Pre-train windows: {pretrain_windows.shape}")

    auto_max_train_length(method_kwargs, window_size)
    trainer = get_trainer(method=method, input_dims=D, device=device,
                          dataset=dataset, **method_kwargs)
    eval_results = []

    def encode_and_score(tr, ep, row_method, use_target):
        full_repr = encode_sequence(tr, np.concatenate([train_x, test_x], axis=0),
                                    use_target=use_target)
        if full_repr.ndim == 1:
            return False
        train_repr = full_repr[:T_train]
        test_repr = full_repr[T_train:T_train + T_test]
        for scoring in scorings:
            train_scores, test_scores = SCORERS[scoring](train_repr, test_repr)
            results = eval_mv_protocols(train_scores, test_scores, test_label,
                                        anomaly_ratio, delay, extra_delays)
            for protocol, d, m in results:
                eval_results.append({
                    'method': row_method,
                    'dataset': dataset,
                    'seed': seed,
                    'epoch': ep,
                    'hidden_dims': hidden_dims,
                    'repr_dims': repr_dims,
                    'scoring': scoring,
                    'protocol': protocol,
                    'delay': d,
                    'f1': round(m['f1'], 4),
                    'precision': round(m['precision'], 4),
                    'recall': round(m['recall'], 4),
                })
            summary = '  '.join(f"{p}@{d}={m['f1']:.4f}" for p, d, m in results)
            main_print(f"    [{row_method} | {scoring}] {summary}")
        return True

    def evaluate(tr, ep):
        main_print(f"\n  [Checkpoint] Encoding and scoring at epoch {ep}...")
        _evaluate_both_encoders(tr, ep, method, encode_and_score)

    loss_csv = f'results/{method}/anomaly/{dataset}/loss_logs/seed{seed}_loss.csv'
    train_time = train_with_checkpoints(trainer, pretrain_windows, checkpoint_epochs, evaluate,
                                        loss_csv)
    append_csv_rows(eval_csv, MV_FIELDNAMES, eval_results)
    main_print(f"  Eval results: {eval_csv}")
    main_print(f"  Total time: {train_time:.1f}s")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection for Time Series SSL')
    parser.add_argument('--method', type=str, default='ts2vec')
    parser.add_argument('--dataset', type=str, default='kpi',
                        help='kpi, yahoo (univariate) or SMD, MSL, SMAP, PSM (multivariate)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of seeds (42, 43, ...)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds to run (e.g. "44"); overrides --iterations.')
    parser.add_argument('--scoring', type=str, default=None,
                        choices=['knn', 'gaussian', 'both'],
                        help='Anomaly scoring method. Default: knn (univariate), '
                             'both (multivariate; one training, both scorings).')
    parser.add_argument('--checkpoint_epochs', type=str, default='50,100,200',
                        help='Comma-separated checkpoint epochs')
    parser.add_argument('--hidden_dims', type=int, default=None)
    parser.add_argument('--output_dims', type=int, default=None)
    parser.add_argument('--eval_csv', type=str, default=None,
                        help=f'Output CSV path (default: {ANO_CSV} for univariate, '
                             f'{ANO_MV_CSV} for multivariate).')
    parser.add_argument('--delay', type=int, default=MV_DELAY,
                        help=f'Delay tolerance for multivariate datasets (default {MV_DELAY}). '
                             'Univariate delays come from the dataset pickle.')
    parser.add_argument('--extra_delays', type=str,
                        default=','.join(str(d) for d in MV_EXTRA_DELAYS),
                        help='Additional delay tolerances recorded for the delay-dependent '
                             'multivariate protocols (sensitivity check). "" disables.')
    parser.add_argument('--window_size', type=int, default=100,
                        help='Sliding window length for multivariate datasets only (default 100, '
                             'Anomaly-Transformer convention).')
    parser.add_argument('--anomaly_ratio', type=float, default=None,
                        help='Top-fraction threshold for multivariate datasets only. '
                             'Defaults to MV_ANOMALY_RATIO[dataset] (SMD=0.005, others=0.01).')
    args = parser.parse_args()

    is_mv = args.dataset in ANOMALY_MV_DATASETS
    scoring = args.scoring or ('both' if is_mv else 'knn')
    if scoring == 'both' and not is_mv:
        parser.error('--scoring both is only supported for multivariate datasets')
    eval_csv = args.eval_csv or (ANO_MV_CSV if is_mv else ANO_CSV)

    checkpoint_epochs = parse_epochs(args.checkpoint_epochs)
    method_kwargs = dim_overrides(args.hidden_dims, args.output_dims)
    hidden_dims, repr_dims = dims_for(args.method, args.hidden_dims, args.output_dims)
    seeds = ([int(s) for s in args.seeds.split(',')] if args.seeds
             else default_seeds(args.iterations))

    rank, world_size, local_rank = setup_ddp()
    device = get_device(local_rank)

    if is_mv:
        run_fn = run_anomaly_mv_checkpoints
        run_kwargs = dict(
            scorings=['knn', 'gaussian'] if scoring == 'both' else [scoring],
            window_size=args.window_size, anomaly_ratio=args.anomaly_ratio,
            delay=args.delay,
            extra_delays=tuple(int(x) for x in args.extra_delays.split(',') if x.strip()),
        )
    else:
        run_fn = run_anomaly_checkpoints
        run_kwargs = dict(scoring=scoring)

    failures = FailureLog(rank)
    for iteration, seed in enumerate(seeds, start=1):
        # Seed-level sharding: one dataset per invocation means seeds are the
        # only parallelizable axis. Without this, every torchrun rank trains
        # the same (method, dataset, seed) and appends duplicate CSV rows.
        if (iteration - 1) % world_size != rank:
            continue

        print(f"\n[GPU {rank}] >>> ITERATION {iteration}/{len(seeds)} | Seed: {seed}")
        if ano_done(eval_csv, args.method, args.dataset, seed, max(checkpoint_epochs),
                    scoring, hidden_dims, repr_dims):
            print(f"  [GPU {rank}] [Skip] {args.dataset} (seed {seed}, {scoring} already done)")
            continue

        failures.run(
            f'{args.dataset} (seed {seed})', run_fn,
            method=args.method, dataset=args.dataset, checkpoint_epochs=checkpoint_epochs,
            device=device, seed=seed, eval_csv=eval_csv, **run_kwargs, **method_kwargs,
        )

    main_print(f"\n{'='*70}")
    main_print(f"{args.method.upper()} ANOMALY DETECTION COMPLETE")
    main_print(f"{'='*70}")
    failures.finish()


if __name__ == '__main__':
    main()
