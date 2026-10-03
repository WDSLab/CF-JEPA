"""
Dataset-parallel Anomaly Detection Script for Time Series SSL Baselines

Approach:
    1. Train SSL encoder on normal (training) data
    2. Encode train+test series → per-timestep representations
    3. Score anomalies using representation-based methods (knn, gaussian)
    4. Threshold on training scores (mean + 4*std)
    5. Evaluate with delay-adjusted F1/Precision/Recall
       (multivariate: same protocol plus comparison variants, see eval_mv_protocols)

Usage:
    python scripts/benchmarks/run_anomaly.py --method ts2vec --dataset kpi --epochs 200 --iterations 3
    torchrun --nproc_per_node=2 scripts/benchmarks/run_anomaly.py --method ts2vec --dataset kpi --epochs 200
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baselines import get_trainer, get_method_dims
from utils.data_loaders import (
    load_anomaly, gen_ano_train_data,
    load_anomaly_mv, gen_ano_mv_train_data, ANOMALY_MV_DATASETS,
)


# =============================================================================
# DDP Helpers
# =============================================================================

def setup_ddp():
    if 'RANK' in os.environ:
        backend = 'gloo'
        dist.init_process_group(backend=backend)
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
# Encoding
# =============================================================================

def encode_series(trainer, series_1d, max_chunk=5000, overlap=200, min_len=210,
                  use_target=False):
    """Encode a single 1D series into per-timestep representations.

    For long series, encodes in overlapping chunks and stitches results.
    Returns shape (T, D) for methods supporting raw=True,
    or (D,) for pooled-only methods (e.g., SimMTM).

    `use_target=True` selects the EMA target encoder for trainers that support
    it (CF-JEPA family); other trainers ignore the kwarg.
    """
    T = len(series_1d)
    data = series_1d.reshape(1, T, 1).astype(np.float32)

    # Some trainers don't accept use_target (older API). Probe once to decide.
    def _enc(arr, raw):
        if use_target:
            try:
                return trainer.encode(arr, raw=raw, use_target=True)
            except TypeError:
                return trainer.encode(arr, raw=raw)
        return trainer.encode(arr, raw=raw)

    # Pad short series
    if T < min_len:
        pad = np.zeros((1, min_len - T, 1), dtype=np.float32)
        data = np.concatenate([pad, data], axis=1)

    if T <= max_chunk:
        z = _enc(data, raw=True)
        if z.ndim == 3:
            z = z[0]  # (T_enc, D)
            # If padded, strip prefix
            if T < min_len:
                z = z[-(T):]
            # Handle patch-based methods where T_enc != T
            if z.shape[0] != T:
                z = _interpolate_repr(z, T)
            return z
        else:
            return z.squeeze()  # (D,) — pooled method

    # Chunked encoding for long series
    chunks = []
    for start in range(0, T, max_chunk - overlap):
        end = min(start + max_chunk, T)
        chunk_data = series_1d[start:end].reshape(1, -1, 1).astype(np.float32)

        # Pad short chunks
        chunk_len = chunk_data.shape[1]
        if chunk_len < min_len:
            pad = np.zeros((1, min_len - chunk_len, 1), dtype=np.float32)
            chunk_data = np.concatenate([pad, chunk_data], axis=1)

        z = _enc(chunk_data, raw=True)
        if z.ndim == 3:
            z = z[0]
            # Strip padding
            if chunk_len < min_len:
                z = z[-(chunk_len):]
            # Handle patch-based: interpolate to chunk length
            if z.shape[0] != (end - start):
                z = _interpolate_repr(z, end - start)

            if start == 0:
                chunks.append(z)
            else:
                # Skip overlap region
                chunks.append(z[overlap:])
        else:
            # Pooled method — can't do per-timestep with chunking
            return _enc(
                series_1d.reshape(1, T, 1).astype(np.float32), raw=False
            ).squeeze()

        if end >= T:
            break

    return np.concatenate(chunks, axis=0)[:T]  # ensure exact length


def _interpolate_repr(z, target_len):
    """Linearly interpolate representations to match target length.
    Used for patch-based methods (TS-JEPA) where output T != input T.
    """
    from scipy.interpolate import interp1d
    src_len, D = z.shape
    if src_len == target_len:
        return z
    x_src = np.linspace(0, 1, src_len)
    x_tgt = np.linspace(0, 1, target_len)
    f = interp1d(x_src, z, axis=0, kind='linear')
    return f(x_tgt)


# =============================================================================
# Anomaly Scoring
# =============================================================================

def score_knn(train_repr, test_repr, k=5, max_train=50000):
    """Score anomalies using k-NN distance to training representations.

    For each test timestep, score = distance to k-th nearest training timestep.
    Subsamples training set if too large for efficiency.
    """
    if len(train_repr.shape) == 1:
        # Pooled method — single vector, can't do per-timestep
        return np.zeros(1), np.zeros(1)

    # Subsample training representations for efficiency
    if len(train_repr) > max_train:
        indices = np.random.choice(len(train_repr), max_train, replace=False)
        train_sub = train_repr[indices]
    else:
        train_sub = train_repr

    # Standardize
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_sub)
    test_scaled = scaler.transform(test_repr)

    # Also scale full training set for training scores
    train_full_scaled = scaler.transform(train_repr)

    nn = NearestNeighbors(n_neighbors=min(k, len(train_scaled)), metric='euclidean',
                          algorithm='auto', n_jobs=2)
    nn.fit(train_scaled)

    # Training scores (for threshold estimation)
    train_dist, _ = nn.kneighbors(train_full_scaled)
    train_scores = train_dist[:, -1]

    # Test scores
    test_dist, _ = nn.kneighbors(test_scaled)
    test_scores = test_dist[:, -1]

    return train_scores, test_scores


def score_gaussian(train_repr, test_repr, max_train=50000):
    """Score anomalies using Mahalanobis-like distance.

    Standardizes per-dimension, then computes L2 distance to mean.
    """
    if len(train_repr.shape) == 1:
        return np.zeros(1), np.zeros(1)

    # Subsample for statistics
    if len(train_repr) > max_train:
        indices = np.random.choice(len(train_repr), max_train, replace=False)
        train_sub = train_repr[indices]
    else:
        train_sub = train_repr

    scaler = StandardScaler()
    scaler.fit(train_sub)

    train_scaled = scaler.transform(train_repr)
    test_scaled = scaler.transform(test_repr)

    # L2 distance to origin (=mean after standardization)
    train_scores = np.sqrt(np.sum(train_scaled ** 2, axis=1))
    test_scores = np.sqrt(np.sum(test_scaled ** 2, axis=1))

    return train_scores, test_scores


SCORING_METHODS = {
    'knn': score_knn,
    'gaussian': score_gaussian,
}


# =============================================================================
# Evaluation (delay-adjusted F1)
# =============================================================================

def get_range_proba(predict, label, delay=7):
    """Apply delay tolerance to predictions within anomaly ranges."""
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos:sp] = 1
            else:
                new_predict[pos:sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:
        if 1 in predict[pos:min(pos + delay + 1, sp)]:
            new_predict[pos:sp] = 1
        else:
            new_predict[pos:sp] = 0

    return new_predict


def reconstruct_label(timestamp, label):
    """Reconstruct labels for irregular timestamps (fill gaps with 0)."""
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)
    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval
    new_label = np.zeros(
        shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,),
        dtype=np.int64
    )
    new_label[idx] = label
    return new_label


def eval_ad_result(test_pred_list, test_labels_list, test_timestamps_list, delay):
    """Compute delay-adjusted F1, Precision, Recall across all series."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    labels = []
    pred = []
    for test_pred, test_labels, test_timestamps in zip(
        test_pred_list, test_labels_list, test_timestamps_list
    ):
        assert test_pred.shape == test_labels.shape == test_timestamps.shape
        test_labels_r = reconstruct_label(test_timestamps, test_labels)
        test_pred_r = reconstruct_label(test_timestamps, test_pred)
        test_pred_r = get_range_proba(test_pred_r, test_labels_r, delay)
        labels.append(test_labels_r)
        pred.append(test_pred_r)

    labels = np.concatenate(labels)
    pred = np.concatenate(pred)

    if labels.sum() == 0 or pred.sum() == 0:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    return {
        'f1': f1_score(labels, pred),
        'precision': precision_score(labels, pred),
        'recall': recall_score(labels, pred),
    }


# =============================================================================
# Multivariate encoding + Point-Adjust F1 (SMD/MSL/SMAP/PSM)
# =============================================================================

def encode_long_sequence_mv(trainer, x_2d, max_chunk=5000, overlap=200, min_len=210,
                            use_target=False):
    """Encode a (T, D) multivariate sequence into per-timestep representations.

    Mirrors `encode_series` but handles multivariate input. Chunks long sequences
    with overlap to keep GPU memory bounded.
    """
    T, D = x_2d.shape
    data = x_2d.reshape(1, T, D).astype(np.float32)

    def _enc(arr, raw):
        if use_target:
            try:
                return trainer.encode(arr, raw=raw, use_target=True)
            except TypeError:
                return trainer.encode(arr, raw=raw)
        return trainer.encode(arr, raw=raw)

    if T < min_len:
        pad = np.zeros((1, min_len - T, D), dtype=np.float32)
        data = np.concatenate([pad, data], axis=1)

    if T <= max_chunk:
        z = _enc(data, raw=True)
        if z.ndim == 3:
            z = z[0]
            if T < min_len:
                z = z[-T:]
            if z.shape[0] != T:
                z = _interpolate_repr(z, T)
            return z
        return z.squeeze()  # pooled-only method (e.g. SimMTM)

    chunks = []
    for start in range(0, T, max_chunk - overlap):
        end = min(start + max_chunk, T)
        chunk = x_2d[start:end].reshape(1, -1, D).astype(np.float32)
        chunk_len = chunk.shape[1]
        if chunk_len < min_len:
            pad = np.zeros((1, min_len - chunk_len, D), dtype=np.float32)
            chunk = np.concatenate([pad, chunk], axis=1)

        z = _enc(chunk, raw=True)
        if z.ndim == 3:
            z = z[0]
            if chunk_len < min_len:
                z = z[-chunk_len:]
            if z.shape[0] != (end - start):
                z = _interpolate_repr(z, end - start)
            if start == 0:
                chunks.append(z)
            else:
                chunks.append(z[overlap:])
        else:
            return _enc(x_2d.reshape(1, T, D).astype(np.float32), raw=False).squeeze()

        if end >= T:
            break

    return np.concatenate(chunks, axis=0)[:T]


def eval_pa_f1(test_pred, test_label):
    """Point-adjust F1 (standard for SMD/MSL/SMAP/PSM).

    Convention from OmniAnomaly: if any timestep within a ground-truth anomaly
    segment is predicted as positive, the entire segment is marked as detected.
    Returns dict with f1 / precision / recall (and raw counterparts for sanity).
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    pred = np.asarray(test_pred).astype(np.int64).copy()
    label = np.asarray(test_label).astype(np.int64)
    anomaly_state = False
    for i in range(len(label)):
        if label[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if label[j] == 0:
                    break
                pred[j] = 1
            for j in range(i, len(label)):
                if label[j] == 0:
                    break
                pred[j] = 1
        elif label[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    if label.sum() == 0 or pred.sum() == 0:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    return {
        'f1': f1_score(label, pred),
        'precision': precision_score(label, pred),
        'recall': recall_score(label, pred),
    }


def _threshold_by_ratio(scores, ratio):
    """Return threshold so that the top `ratio` fraction of scores are anomalies."""
    return float(np.quantile(scores, 1.0 - ratio))


def _suppress_within_delay(pred, delay):
    """Drop a detection if another one fired within the preceding `delay` steps.

    Same sequential rule as the univariate path (`run_anomaly_checkpoint`).
    """
    pred = pred.copy()
    for i in range(len(pred)):
        if i >= delay and pred[i - delay:i].sum() >= 1:
            pred[i] = 0
    return pred


def _prf(pred, label):
    from sklearn.metrics import f1_score, precision_score, recall_score
    if label.sum() == 0 or pred.sum() == 0:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    return {
        'f1': f1_score(label, pred),
        'precision': precision_score(label, pred),
        'recall': recall_score(label, pred),
    }


def eval_mv_protocols(train_scores, test_scores, test_label, anomaly_ratio, delay,
                      extra_delays=()):
    """Score one (T_test,) sequence under every multivariate protocol.

    Point adjustment overstates performance (Kim et al., AAAI 2022), so the
    multivariate benchmarks are evaluated with the univariate protocol
    (`delay_train4sigma`, used in the paper). All variants come from the same
    scores, so no retraining is needed to compare protocols.

        delay_train4sigma  univariate protocol: train mean+4std threshold,
                           detection suppression, delay-adjusted F1
        delay_ratio        top-`anomaly_ratio` threshold, then the same delay step
        point_train4sigma  train mean+4std threshold, unadjusted point-wise F1
        point_ratio        top-`anomaly_ratio` threshold, unadjusted point-wise F1
        pa_ratio           previous paper protocol (point-adjust); sanity check only

    Returns a list of (protocol, delay, metrics). All five protocols are scored at
    the primary `delay`; the two delay-dependent ones are also scored at each of
    `extra_delays` (delay-sensitivity check, e.g. Yahoo's 3).
    """
    label = np.asarray(test_label).astype(np.int64)
    thr_sigma = np.mean(train_scores) + 4 * np.std(train_scores)
    thr_ratio = _threshold_by_ratio(np.concatenate([train_scores, test_scores]),
                                    anomaly_ratio)
    pred_sigma = (test_scores > thr_sigma).astype(np.int64)
    pred_ratio = (test_scores > thr_ratio).astype(np.int64)

    def _delay(pred, d):
        return get_range_proba(_suppress_within_delay(pred, d), label, d)

    out = [
        ('delay_train4sigma', delay, _prf(_delay(pred_sigma, delay), label)),
        ('delay_ratio', delay, _prf(_delay(pred_ratio, delay), label)),
        ('point_train4sigma', delay, _prf(pred_sigma, label)),
        ('point_ratio', delay, _prf(pred_ratio, label)),
        ('pa_ratio', delay, eval_pa_f1(pred_ratio, label)),
    ]
    for d in extra_delays:
        if d == delay:
            continue
        out.append(('delay_train4sigma', d, _prf(_delay(pred_sigma, d), label)))
        out.append(('delay_ratio', d, _prf(_delay(pred_ratio, d), label)))
    return out


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

# Multivariate results go to their own CSV (with a `protocol` column) so the
# univariate / crop rows in results/ano_results.csv are never mixed in.
MV_EVAL_CSV = 'results/ano_mv_results.csv'
MV_FIELDNAMES = ['method', 'dataset', 'seed', 'epoch', 'hidden_dims', 'repr_dims',
                 'scoring', 'protocol', 'delay', 'f1', 'precision', 'recall']


# =============================================================================
# Main Pipeline
# =============================================================================

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_anomaly_checkpoint(method, dataset_name, max_epochs=200,
                           checkpoint_epochs=None, device='cuda',
                           local_rank=-1, seed=42, hidden_dims=None,
                           repr_dims=None, scoring='knn', eval_csv=None,
                           **method_kwargs):
    """Train encoder and evaluate anomaly detection at each checkpoint.

    1. Load anomaly dataset (dict of variable-length series)
    2. Pre-train encoder on all training series
    3. At each checkpoint: encode per-series → score → threshold → F1
    4. Save results to CSV
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = [50, 100, 200]

    set_seed(seed)
    rank0 = is_main_process()

    if rank0:
        print(f"\n{'='*60}")
        print(f"Anomaly Detection: {dataset_name}")
        print(f"Method: {method} | Scoring: {scoring}")
        print(f"Checkpoints: {checkpoint_epochs}")
        print(f"Seed: {seed} | Device: {device}")
        print(f"Hidden dims: {hidden_dims} | Repr dims: {repr_dims}")
        print(f"{'='*60}")

    # Load data
    all_train_data, all_train_labels, all_train_timestamps, \
        all_test_data, all_test_labels, all_test_timestamps, delay = load_anomaly(dataset_name)

    series_keys = sorted(all_train_data.keys())
    if rank0:
        print(f"  Series count: {len(series_keys)}")
        print(f"  Delay: {delay}")
        total_train = sum(len(all_train_data[k]) for k in series_keys)
        total_test = sum(len(all_test_data[k]) for k in series_keys)
        print(f"  Total train points: {total_train:,}")
        print(f"  Total test points: {total_test:,}")

    # Prepare pre-training data
    pretrain_data = gen_ano_train_data(all_train_data)  # (n_series, max_T, 1)
    input_dims = 1
    T = pretrain_data.shape[1]

    if rank0:
        print(f"  Pre-train data shape: {pretrain_data.shape}")

    # Auto max_train_length for long sequences (same as classification/forecasting)
    if 'max_train_length' not in method_kwargs and T > 2048:
        method_kwargs['max_train_length'] = 2048
        if rank0:
            print(f"  Auto max_train_length=2048 (T={T})")

    # Create trainer
    trainer = get_trainer(method=method, input_dims=input_dims, device=device,
                          dataset=dataset_name, **method_kwargs)

    # Results CSV
    if eval_csv is None:
        eval_csv = 'results/ano_results.csv'
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

    scoring_fn = SCORING_METHODS[scoring]

    def _epoch_callback(tr, ep, loss):
        loss_log.append((ep, loss))
        if ep not in checkpoint_set:
            return

        if rank0:
            print(f"\n  [Checkpoint] Encoding and scoring at epoch {ep}...")

        test_pred_list = []
        test_labels_list = []
        test_timestamps_list = []

        for ki, k in enumerate(series_keys):
            train_series = all_train_data[k]
            test_series = all_test_data[k]

            full_series = np.concatenate([train_series, test_series])
            full_repr = encode_series(tr, full_series)

            if full_repr.ndim == 1:
                if rank0 and ki == 0:
                    print(f"    Warning: {method} does not support per-timestep encoding, skipping")
                break

            train_repr = full_repr[:len(train_series)]
            test_repr = full_repr[len(train_series):]

            train_scores, test_scores = scoring_fn(train_repr, test_repr)

            thr = np.mean(train_scores) + 4 * np.std(train_scores)
            test_pred = (test_scores > thr).astype(np.int64)

            for i in range(len(test_pred)):
                if i >= delay and test_pred[i - delay:i].sum() >= 1:
                    test_pred[i] = 0

            test_pred_list.append(test_pred)
            test_labels_list.append(all_test_labels[k])
            test_timestamps_list.append(all_test_timestamps[k])

            if rank0 and (ki + 1) % 50 == 0:
                print(f"    Processed {ki+1}/{len(series_keys)} series")

        if test_pred_list:
            metrics = eval_ad_result(
                test_pred_list, test_labels_list, test_timestamps_list, delay
            )
            eval_results.append({
                'method': method,
                'dataset': dataset_name,
                'seed': seed,
                'epoch': ep,
                'hidden_dims': hidden_dims,
                'repr_dims': repr_dims,
                'scoring': scoring,
                'f1': round(metrics['f1'], 4),
                'precision': round(metrics['precision'], 4),
                'recall': round(metrics['recall'], 4),
            })
            if rank0:
                print(f"    F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")
        else:
            if rank0:
                print(f"    Skipped evaluation (no per-timestep representations)")

        # Dual-encoder eval: for cf_jepa, also evaluate with the EMA target
        # encoder.
        # Saved as method=f'{method}_target'. Same training, just extra
        # encode + score passes — no extra GPU training time.
        if (method.startswith('cf_jepa')
                and hasattr(tr, 'target_encoder')
                and tr.target_encoder is not None
                and test_pred_list):  # skip if main eval was skipped (pooled-only)
            if rank0:
                print(f"  [Dual eval] Re-encoding with target encoder...")
            t_pred_list = []
            t_labels_list = []
            t_timestamps_list = []
            for ki, k in enumerate(series_keys):
                train_series = all_train_data[k]
                test_series = all_test_data[k]
                full_series = np.concatenate([train_series, test_series])
                full_repr_t = encode_series(tr, full_series, use_target=True)
                if full_repr_t.ndim == 1:
                    break
                train_repr_t = full_repr_t[:len(train_series)]
                test_repr_t = full_repr_t[len(train_series):]
                train_scores_t, test_scores_t = scoring_fn(train_repr_t, test_repr_t)
                thr_t = np.mean(train_scores_t) + 4 * np.std(train_scores_t)
                test_pred_t = (test_scores_t > thr_t).astype(np.int64)
                for i in range(len(test_pred_t)):
                    if i >= delay and test_pred_t[i - delay:i].sum() >= 1:
                        test_pred_t[i] = 0
                t_pred_list.append(test_pred_t)
                t_labels_list.append(all_test_labels[k])
                t_timestamps_list.append(all_test_timestamps[k])
            if t_pred_list:
                metrics_t = eval_ad_result(
                    t_pred_list, t_labels_list, t_timestamps_list, delay
                )
                eval_results.append({
                    'method': f'{method}_target',
                    'dataset': dataset_name,
                    'seed': seed,
                    'epoch': ep,
                    'hidden_dims': hidden_dims,
                    'repr_dims': repr_dims,
                    'scoring': scoring,
                    'f1': round(metrics_t['f1'], 4),
                    'precision': round(metrics_t['precision'], 4),
                    'recall': round(metrics_t['recall'], 4),
                })
                if rank0:
                    print(f"    [target] F1={metrics_t['f1']:.4f}  P={metrics_t['precision']:.4f}  R={metrics_t['recall']:.4f}")

        import gc; gc.collect()

    if rank0:
        print(f"\n  Training 1 -> {max_checkpoint} (continuous)...")
    trainer.fit(pretrain_data, epochs=max_checkpoint, verbose=False,
                callback=_epoch_callback)

    train_time = time.time() - t_start

    # Append eval results to CSV
    if eval_results:
        fieldnames = ['method', 'dataset', 'seed', 'epoch', 'hidden_dims', 'repr_dims',
                      'scoring', 'f1', 'precision', 'recall']
        file_exists = os.path.exists(eval_csv)
        with open(eval_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(eval_results)

    # Save loss log
    loss_dir = f'results/{method}/anomaly/{dataset_name}/loss_logs'
    os.makedirs(loss_dir, exist_ok=True)
    loss_csv = f'{loss_dir}/seed{seed}_loss.csv'
    with open(loss_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss'])
        writer.writeheader()
        for ep, loss in loss_log:
            writer.writerow({'epoch': ep, 'loss': round(loss, 6)})

    if rank0:
        print(f"\n  Loss log: {loss_csv}")
        print(f"  Eval results: {eval_csv}")
        print(f"  Total time: {train_time:.1f}s")

    return train_time


def run_anomaly_mv_checkpoint(method, dataset_name, max_epochs=200,
                              checkpoint_epochs=None, device='cuda',
                              local_rank=-1, seed=42, hidden_dims=None,
                              repr_dims=None, scorings=('knn', 'gaussian'),
                              eval_csv=None, window_size=100, anomaly_ratio=None,
                              delay=MV_DELAY, extra_delays=MV_EXTRA_DELAYS,
                              **method_kwargs):
    """Multivariate anomaly pipeline (SMD / MSL / SMAP / PSM).

    Pipeline:
        1. Load (T_train, D) + (T_test, D) + (T_test,) label
        2. Split train into non-overlapping windows of `window_size`
        3. Pre-train encoder on windows
        4. At each checkpoint: encode train+test once, then for every scoring
           function write one row per protocol (see `eval_mv_protocols`)
        5. Save to MV_EVAL_CSV; auto-emit f'{method}_target' rows for cf_jepa

    All scorings share one training run (the encoder does not depend on the score).
    `anomaly_ratio` defaults to `MV_ANOMALY_RATIO[dataset_name]` (Anomaly-Transformer
    convention) and only affects the *_ratio protocols.
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = [50, 100, 200]
    if anomaly_ratio is None:
        anomaly_ratio = MV_ANOMALY_RATIO.get(dataset_name, 0.01)
    scorings = list(scorings)

    set_seed(seed)
    rank0 = is_main_process()

    if rank0:
        print(f"\n{'='*60}")
        print(f"Anomaly Detection (multivariate): {dataset_name}")
        print(f"Method: {method} | Scoring: {scorings}")
        print(f"Checkpoints: {checkpoint_epochs}")
        print(f"Seed: {seed} | Device: {device}")
        print(f"Hidden dims: {hidden_dims} | Repr dims: {repr_dims}")
        print(f"Window: {window_size} | Anomaly ratio: {anomaly_ratio} | Delay: {delay}")
        print(f"{'='*60}")

    train_x, test_x, test_label = load_anomaly_mv(dataset_name)
    T_train, D = train_x.shape
    T_test = test_x.shape[0]
    if rank0:
        print(f"  Train: ({T_train}, {D})  Test: ({T_test}, {D})  "
              f"anomaly={test_label.mean():.4f}")

    pretrain_windows = gen_ano_mv_train_data(train_x, window_size=window_size,
                                             stride=window_size)
    if rank0:
        print(f"  Pre-train windows: {pretrain_windows.shape}")

    input_dims = D
    if 'max_train_length' not in method_kwargs and window_size > 2048:
        method_kwargs['max_train_length'] = 2048

    trainer = get_trainer(method=method, input_dims=input_dims, device=device,
                          dataset=dataset_name, **method_kwargs)

    if eval_csv is None:
        eval_csv = MV_EVAL_CSV
    os.makedirs(os.path.dirname(eval_csv) or '.', exist_ok=True)

    loss_log = []
    eval_results = []
    t_start = time.time()

    sorted_checkpoints = sorted(ep for ep in checkpoint_epochs if ep <= max_epochs)
    if not sorted_checkpoints:
        return 0.0
    max_checkpoint = sorted_checkpoints[-1]
    checkpoint_set = set(sorted_checkpoints)

    def _score_and_eval(tr, ep, row_method, use_target=False):
        """Encode train+test once; append one row per (scoring, protocol)."""
        full = np.concatenate([train_x, test_x], axis=0)
        full_repr = encode_long_sequence_mv(tr, full, use_target=use_target)
        if full_repr.ndim == 1:
            return False  # pooled-only method (e.g. SimMTM) — skip
        train_repr = full_repr[:T_train]
        test_repr = full_repr[T_train:T_train + T_test]
        for scoring in scorings:
            train_scores, test_scores = SCORING_METHODS[scoring](train_repr, test_repr)
            results = eval_mv_protocols(train_scores, test_scores, test_label,
                                        anomaly_ratio, delay, extra_delays)
            for protocol, d, m in results:
                eval_results.append({
                    'method': row_method,
                    'dataset': dataset_name,
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
            if rank0:
                summary = '  '.join(f"{p}@{d}={m['f1']:.4f}" for p, d, m in results)
                print(f"    [{row_method} | {scoring}] {summary}")
        return True

    def _epoch_callback(tr, ep, loss):
        loss_log.append((ep, loss))
        if ep not in checkpoint_set:
            return
        if rank0:
            print(f"\n  [Checkpoint] Encoding and scoring at epoch {ep}...")

        if not _score_and_eval(tr, ep, method, use_target=False):
            if rank0:
                print(f"    Skipped (pooled-only method)")
            return

        # Dual-encoder eval — mirror of the univariate path.
        if (method.startswith('cf_jepa')
                and hasattr(tr, 'target_encoder')
                and tr.target_encoder is not None):
            if rank0:
                print(f"  [Dual eval] Re-encoding with target encoder...")
            _score_and_eval(tr, ep, f'{method}_target', use_target=True)

        import gc; gc.collect()

    if rank0:
        print(f"\n  Training 1 -> {max_checkpoint} (continuous)...")
    trainer.fit(pretrain_windows, epochs=max_checkpoint, verbose=False,
                callback=_epoch_callback)

    train_time = time.time() - t_start

    if eval_results:
        file_exists = os.path.exists(eval_csv)
        with open(eval_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=MV_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerows(eval_results)

    loss_dir = f'results/{method}/anomaly/{dataset_name}/loss_logs'
    os.makedirs(loss_dir, exist_ok=True)
    loss_csv = f'{loss_dir}/seed{seed}_loss.csv'
    with open(loss_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss'])
        writer.writeheader()
        for ep, loss in loss_log:
            writer.writerow({'epoch': ep, 'loss': round(loss, 6)})

    if rank0:
        print(f"\n  Loss log: {loss_csv}")
        print(f"  Eval results: {eval_csv}")
        print(f"  Total time: {train_time:.1f}s")

    return train_time


def _is_anomaly_done(method, dataset, seed, max_epoch, scoring='knn',
                     hidden_dims=None, repr_dims=None,
                     eval_csv='results/ano_results.csv'):
    """Check if last checkpoint results exist in anomaly CSV.

    `scoring='both'` (multivariate) requires knn and gaussian rows, and only
    counts rows carrying a `protocol` column (i.e. written by the current MV path).
    """
    if not os.path.exists(eval_csv):
        return False
    need = {'knn', 'gaussian'} if scoring == 'both' else {scoring}
    found = set()
    with open(eval_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if scoring == 'both' and not row.get('protocol'):
                continue
            if (row.get('method') == method and
                row.get('dataset') == dataset and
                row.get('seed') == str(seed) and
                row.get('epoch') == str(max_epoch) and
                row.get('hidden_dims', str(hidden_dims)) == str(hidden_dims) and
                row.get('repr_dims', str(repr_dims)) == str(repr_dims) and
                row.get('scoring') in need):
                found.add(row['scoring'])
    return found == need


def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection for Time Series SSL')
    parser.add_argument('--method', type=str, default='ts2vec')
    parser.add_argument('--dataset', type=str, default='kpi',
                        help='Anomaly dataset name (kpi, yahoo)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--iterations', type=int, default=3)
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
                        help='Output CSV path (default: results/ano_results.csv for '
                             f'univariate, {MV_EVAL_CSV} for multivariate).')
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
    if args.scoring is None:
        args.scoring = 'both' if is_mv else 'knn'
    if args.scoring == 'both' and not is_mv:
        parser.error('--scoring both is only supported for multivariate datasets')
    if args.eval_csv is None:
        args.eval_csv = MV_EVAL_CSV if is_mv else 'results/ano_results.csv'

    checkpoint_epochs = [int(x) for x in args.checkpoint_epochs.split(',')]
    max_epoch = max(checkpoint_epochs)

    method_kwargs = {}
    if args.hidden_dims is not None:
        method_kwargs['hidden_dims'] = args.hidden_dims
    if args.output_dims is not None:
        method_kwargs['output_dims'] = args.output_dims

    default_h, default_r = get_method_dims(args.method)
    actual_hidden = args.hidden_dims if args.hidden_dims is not None else default_h
    actual_repr = args.output_dims if args.output_dims is not None else default_r

    rank, world_size, local_rank = setup_ddp()
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    # Anomaly datasets: each dataset is a single unit (not parallelizable like classification)
    # Multi-GPU parallelism applies across methods/seeds, not within a dataset
    seeds = [42, 43, 44]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]
        args.iterations = len(seeds)

    for iteration in range(1, args.iterations + 1):
        seed = seeds[iteration - 1] if iteration <= len(seeds) else 42 + iteration - 1

        # Seed-level sharding: one dataset per invocation means seeds are the
        # only parallelizable axis. Without this, every torchrun rank trains
        # the same (method, dataset, seed) and appends duplicate CSV rows.
        if (iteration - 1) % world_size != rank:
            continue

        print(f"\n[GPU {rank}] >>> ITERATION {iteration}/{args.iterations} | Seed: {seed}")

        if _is_anomaly_done(args.method, args.dataset, seed, max_epoch,
                            scoring=args.scoring,
                            hidden_dims=actual_hidden, repr_dims=actual_repr,
                            eval_csv=args.eval_csv):
            print(f"  [GPU {rank}] [Skip] {args.dataset} (seed {seed}, {args.scoring} already done)")
            continue

        try:
            if is_mv:
                run_anomaly_mv_checkpoint(
                    method=args.method, dataset_name=args.dataset,
                    max_epochs=max_epoch, checkpoint_epochs=checkpoint_epochs,
                    device=device, local_rank=local_rank, seed=seed,
                    hidden_dims=actual_hidden, repr_dims=actual_repr,
                    scorings=(['knn', 'gaussian'] if args.scoring == 'both'
                              else [args.scoring]),
                    eval_csv=args.eval_csv,
                    window_size=args.window_size,
                    anomaly_ratio=args.anomaly_ratio,
                    delay=args.delay,
                    extra_delays=tuple(int(x) for x in args.extra_delays.split(',') if x.strip()),
                    **method_kwargs,
                )
            else:
                run_anomaly_checkpoint(
                    method=args.method, dataset_name=args.dataset,
                    max_epochs=max_epoch, checkpoint_epochs=checkpoint_epochs,
                    device=device, local_rank=local_rank, seed=seed,
                    hidden_dims=actual_hidden, repr_dims=actual_repr,
                    scoring=args.scoring, eval_csv=args.eval_csv,
                    **method_kwargs,
                )
        except Exception as e:
            print(f"Error on {args.dataset}: {e}")
            import traceback
            traceback.print_exc()

    if is_main_process():
        print(f"\n{'='*70}")
        print(f"{args.method.upper()} ANOMALY DETECTION COMPLETE")
        print(f"{'='*70}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
