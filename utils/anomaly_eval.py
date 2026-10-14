"""Anomaly scoring, thresholds and F1 protocols for the anomaly benchmark.

Pure numpy / scikit-learn (no trainer, no torch), so the runner
(scripts/benchmarks/run_anomaly.py) and the unit tests share one copy.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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


SCORERS = {
    'knn': score_knn,
    'gaussian': score_gaussian,
}


# =============================================================================
# Thresholding and evaluation
# =============================================================================

def threshold_train4sigma(train_scores):
    """Training-score threshold mean + 4 std (the univariate protocol)."""
    return np.mean(train_scores) + 4 * np.std(train_scores)


def _threshold_by_ratio(scores, ratio):
    """Return threshold so that the top `ratio` fraction of scores are anomalies."""
    return float(np.quantile(scores, 1.0 - ratio))


def suppress_within_delay(pred, delay):
    """Drop a detection if another one fired within the preceding `delay` steps."""
    pred = pred.copy()
    for i in range(len(pred)):
        if i >= delay and pred[i - delay:i].sum() >= 1:
            pred[i] = 0
    return pred


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


def _prf(pred, label):
    if label.sum() == 0 or pred.sum() == 0:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    return {
        'f1': f1_score(label, pred),
        'precision': precision_score(label, pred),
        'recall': recall_score(label, pred),
    }


def eval_ad_result(test_pred_list, test_labels_list, test_timestamps_list, delay):
    """Compute delay-adjusted F1, Precision, Recall across all series."""
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

    return _prf(np.concatenate(pred), np.concatenate(labels))


def eval_pa_f1(test_pred, test_label):
    """Point-adjust F1 (the previous SMD/MSL/SMAP/PSM protocol).

    Convention from OmniAnomaly: if any timestep within a ground-truth anomaly
    segment is predicted as positive, the entire segment is marked as detected.
    """
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

    return _prf(pred, label)


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
        pa_ratio           point-adjust F1 (common in prior work); comparison only

    Returns a list of (protocol, delay, metrics). All five protocols are scored at
    the primary `delay`; the two delay-dependent ones are also scored at each of
    `extra_delays` (delay-sensitivity check, e.g. Yahoo's 3).
    """
    label = np.asarray(test_label).astype(np.int64)
    thr_ratio = _threshold_by_ratio(np.concatenate([train_scores, test_scores]),
                                    anomaly_ratio)
    pred_sigma = (test_scores > threshold_train4sigma(train_scores)).astype(np.int64)
    pred_ratio = (test_scores > thr_ratio).astype(np.int64)

    def _delay(pred, d):
        return get_range_proba(suppress_within_delay(pred, d), label, d)

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
