"""
Dataset-parallel Classification Script for Time Series SSL Baselines

One continuous training run per (dataset, seed). At each checkpoint epoch the
encoder is frozen, the representations are pooled (maxpool, adaptive_avg) and
scored with SVM heads; one row per (pooling, protocol) goes to the result CSV.
Under torchrun each GPU takes the next unclaimed dataset.

Usage:
    python scripts/benchmarks/run_classification.py --method ts2vec --datasets_file splits/datasets_ucr_t1.txt --iterations 3
    torchrun --nproc_per_node=2 scripts/benchmarks/run_classification.py --method ts2vec --datasets_file splits/datasets_ucr_t1.txt --iterations 3

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

import torch
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baselines import get_trainer
from config import UCR_PATH, UEA_PATH
from utils.data_loaders import load_UCR, load_UEA
from utils.result_status import (
    CLS_CSV, CLS_POOLINGS, CLS_PROTOCOLS, cls_completed_datasets, cls_done_pairs,
    default_seeds, dim_overrides, dims_for,
)
from utils.runner import (
    FailureLog, append_csv_rows, auto_max_train_length, banner, dual_encoder_ready,
    get_device, iter_work, main_print, parse_epochs, set_seed, setup_ddp,
    train_with_checkpoints,
)

POOL_SIZE = 8
CSV_FIELDNAMES = ['method', 'dataset', 'seed', 'epoch', 'hidden_dims', 'repr_dims',
                  'pooling', 'pool_size', 'eval_protocol', 'accuracy']


# =============================================================================
# Evaluation heads
# =============================================================================

def _subsample(features, y, n):
    split = train_test_split(features, y, train_size=n, random_state=0, stratify=y)
    return split[0], split[2]


def fit_svm(features, y, kernel='rbf', MAX_SAMPLES=10000, GRID_SAMPLES=5000):
    """SVM with C chosen by 5-fold grid search (hard margin on tiny datasets).

    If the solver returns non-finite dual coefficients (rare hard datasets such
    as PigCVP: T=2000, N=104, 52 classes), refit with the finite C=1.0.
    """
    svc_kwargs = {'kernel': 'linear'} if kernel == 'linear' else {'gamma': 'scale'}

    def make_svc(**kw):
        return SVC(max_iter=10000000, **svc_kwargs, **kw)

    def fit_with_fallback(C, X, Y):
        try:
            return make_svc(C=C).fit(X, Y)
        except ValueError as e:
            if 'finite' not in str(e).lower():
                raise
            return make_svc(C=1.0).fit(X, Y)

    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if train_size // nb_classes < 5 or train_size < 50:
            return fit_with_fallback(np.inf, features, y)

        grid_search = GridSearchCV(
            make_svc(),
            {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf]},
            cv=5, n_jobs=2
        )
        # Grid search on a subsample for speed; final fit on (at most) MAX_SAMPLES.
        if train_size > GRID_SAMPLES:
            gs_features, gs_y = _subsample(features, y, GRID_SAMPLES)
        else:
            gs_features, gs_y = features, y
        if train_size > MAX_SAMPLES:
            features, y = _subsample(features, y, MAX_SAMPLES)

        grid_search.fit(gs_features, gs_y)
        return fit_with_fallback(grid_search.best_params_['C'], features, y)


def fit_lr(features, y, MAX_SAMPLES=100000):
    if features.shape[0] > MAX_SAMPLES:
        features, y = _subsample(features, y, MAX_SAMPLES)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=0, max_iter=1000000, multi_class='ovr')
    )
    return pipe.fit(features, y)


def fit_knn(features, y):
    pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
    return pipe.fit(features, y)


def evaluate_classifier(z_train, train_labels, z_test, test_labels, protocol='svm'):
    z_train = np.nan_to_num(z_train, nan=0.0, posinf=0.0, neginf=0.0)
    z_test = np.nan_to_num(z_test, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_train)
    z_test = scaler.transform(z_test)
    z_train = np.nan_to_num(z_train, nan=0.0, posinf=0.0, neginf=0.0)
    z_test = np.nan_to_num(z_test, nan=0.0, posinf=0.0, neginf=0.0)
    # Defensive clip: extreme finite values (encoder explosion on hard datasets
    # like PigCVP) survive StandardScaler and cause SVM dual coefs to diverge
    # to Inf during fit. After scaling, normal values lie in [-3, 3]; clipping
    # at +/-10 only touches pathological outliers.
    z_train = np.clip(z_train, -10.0, 10.0)
    z_test = np.clip(z_test, -10.0, 10.0)

    if protocol == 'linear':
        clf = fit_lr(z_train, train_labels)
    elif protocol == 'svm_linear':
        clf = fit_svm(z_train, train_labels, kernel='linear')
    elif protocol == 'knn':
        clf = fit_knn(z_train, train_labels)
    else:
        clf = fit_svm(z_train, train_labels)

    return clf.score(z_test, test_labels)


def _apply_pooling(z, method='maxpool', pool_size=POOL_SIZE):
    """Pool (N, T, D) representations to (N, D')."""
    if z.ndim == 2:
        return z  # Already pooled (e.g., SimMTM)

    N, T, D = z.shape
    z_t = torch.from_numpy(z).float().transpose(1, 2)
    if method == 'maxpool':
        return F.max_pool1d(z_t, kernel_size=T).squeeze(-1).numpy()
    if method == 'adaptive_avg':
        out = F.adaptive_avg_pool1d(z_t, pool_size).transpose(1, 2)
        return out.reshape(N, -1).numpy()
    raise ValueError(f"Unknown pooling: {method}")


# =============================================================================
# Train + evaluate at checkpoints
# =============================================================================

def run_classification_checkpoints(method, dataset, checkpoint_epochs, device='cuda',
                                   data_type='UCR', seed=42, eval_protocols=CLS_PROTOCOLS,
                                   eval_csv=CLS_CSV, **method_kwargs):
    """Train once; at each checkpoint encode, pool and score with the SVM heads.

    (pooling, protocol) rows already in the CSV for this run are not redone.
    """
    set_seed(seed)
    hidden_dims, repr_dims = dims_for(method, method_kwargs.get('hidden_dims'),
                                      method_kwargs.get('output_dims'))
    banner(f"Dataset: {dataset} ({data_type})", f"Method: {method}",
           f"Checkpoints: {checkpoint_epochs}", f"Seed: {seed}", f"Device: {device}",
           f"Hidden dims: {hidden_dims}", f"Repr dims: {repr_dims}")

    loader = load_UEA if data_type == 'UEA' else load_UCR
    train_data, train_labels, test_data, test_labels = loader(dataset)
    main_print(f"Train: {train_data.shape}, Test: {test_data.shape}")
    if train_data.ndim == 2:
        train_data = train_data[:, :, np.newaxis]
        test_data = test_data[:, :, np.newaxis]

    auto_max_train_length(method_kwargs, train_data.shape[1])
    trainer = get_trainer(method=method, input_dims=train_data.shape[2], device=device,
                          dataset=dataset, **method_kwargs)

    # Dual-encoder eval: cf_jepa also reports its EMA target encoder as
    # 'cf_jepa_target'. The main row is the online encoder, which the paper
    # uses for classification (forecasting and anomaly detection use the
    # target encoder; see those runners).
    target_label = 'cf_jepa_target' if method == 'cf_jepa' else None
    done = {label: cls_done_pairs(eval_csv, label, seed, hidden_dims, repr_dims)
            for label in filter(None, (method, target_label))}
    wanted = [(p, proto) for p in CLS_POOLINGS for proto in eval_protocols]
    eval_results = []

    def encode_and_score(tr, ep, row_method, use_target):
        todo = [pair for pair in wanted
                if pair not in done[row_method].get((dataset, ep), set())]
        if not todo:
            return
        enc_kwargs = {'use_target': True} if use_target else {}
        z_train_raw = tr.encode(train_data, raw=True, **enc_kwargs).astype(np.float32)
        z_test_raw = tr.encode(test_data, raw=True, **enc_kwargs).astype(np.float32)
        tag = '' if row_method == method else f'[{row_method}] '
        for pool_method in CLS_POOLINGS:
            z_train = _apply_pooling(z_train_raw, method=pool_method)
            z_test = _apply_pooling(z_test_raw, method=pool_method)
            for protocol in eval_protocols:
                if (pool_method, protocol) not in todo:
                    continue
                acc = evaluate_classifier(z_train, train_labels, z_test, test_labels, protocol)
                eval_results.append({
                    'method': row_method,
                    'dataset': dataset,
                    'seed': seed,
                    'epoch': ep,
                    'hidden_dims': hidden_dims,
                    'repr_dims': repr_dims,
                    'pooling': pool_method,
                    'pool_size': POOL_SIZE if pool_method == 'adaptive_avg' else '',
                    'eval_protocol': protocol,
                    'accuracy': round(acc * 100, 2),
                })
                main_print(f"    {tag}{pool_method} | {protocol}: {acc*100:.2f}%")

    def evaluate(tr, ep):
        main_print(f"\n  [Checkpoint] Encoding at epoch {ep}...")
        encode_and_score(tr, ep, method, use_target=False)
        if target_label and dual_encoder_ready(tr):
            encode_and_score(tr, ep, target_label, use_target=True)

    loss_csv = f'results/{method}/classification/{data_type}/{dataset}/loss_logs/seed{seed}_loss.csv'
    train_time = train_with_checkpoints(trainer, train_data, checkpoint_epochs, evaluate, loss_csv)
    append_csv_rows(eval_csv, CSV_FIELDNAMES, eval_results)
    main_print(f"  Eval results: {eval_csv}")
    main_print(f"  Total train time: {train_time:.1f}s")


# =============================================================================
# Main
# =============================================================================

def _resolve_data_type(ds):
    if os.path.exists(os.path.join(UEA_PATH, ds)):
        return 'UEA'
    if os.path.exists(os.path.join(UCR_PATH, ds)):
        return 'UCR'
    raise SystemExit(f"Dataset {ds!r} not found in {UCR_PATH} or {UEA_PATH}")


def main():
    parser = argparse.ArgumentParser(description='Dataset-parallel Classification for Time Series SSL')
    parser.add_argument('--method', type=str, default='ts2vec',
                        help='ts2vec, softclt, timesurl, trep, autotcl, infots, cost, simmtm, '
                             'ts_jepa, fei, cf_jepa')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--datasets_file', type=str, default=None,
                        help='Text file with dataset names (one per line)')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of seeds (42, 43, ...)')
    parser.add_argument('--checkpoint_epochs', type=str, default='50,100,200',
                        help='Comma-separated checkpoint epochs (default: 50,100,200)')
    parser.add_argument('--eval_protocol', type=str, default='all',
                        choices=['svm', 'svm_linear', 'linear', 'knn', 'all'],
                        help='Evaluation head: all (default, svm+svm_linear) or a single one')
    parser.add_argument('--hidden_dims', type=int, default=None,
                        help='Override hidden dims (default: per-method)')
    parser.add_argument('--output_dims', type=int, default=None,
                        help='Override output/repr dims (default: per-method)')
    parser.add_argument('--eval_csv', type=str, default=CLS_CSV,
                        help=f'Output CSV path (default: {CLS_CSV}). Use a separate path '
                             'for verification runs to avoid contaminating main results.')
    args = parser.parse_args()

    eval_protocols = CLS_PROTOCOLS if args.eval_protocol == 'all' else (args.eval_protocol,)
    checkpoint_epochs = parse_epochs(args.checkpoint_epochs)
    method_kwargs = dim_overrides(args.hidden_dims, args.output_dims)
    hidden_dims, repr_dims = dims_for(args.method, args.hidden_dims, args.output_dims)

    rank, world_size, local_rank = setup_ddp()
    device = get_device(local_rank)

    if args.datasets_file:
        with open(args.datasets_file) as f:
            all_datasets = [line.strip() for line in f if line.strip()]
    elif args.dataset:
        all_datasets = [args.dataset]
    else:
        raise SystemExit("Must specify --datasets_file or --dataset")
    # UCR/UEA is detected per dataset, so one file may mix both.
    data_type_map = {ds: _resolve_data_type(ds) for ds in all_datasets}

    main_print(f"Method: {args.method}")
    main_print(f"World size: {world_size}")
    main_print(f"Iterations: {args.iterations}")
    main_print(f"Datasets: {len(all_datasets)}, checkpoints={checkpoint_epochs}")
    if method_kwargs:
        main_print(f"Override dims: hidden={hidden_dims}, repr={repr_dims}")

    def completed(seed):
        return cls_completed_datasets(args.eval_csv, args.method, seed, checkpoint_epochs,
                                      hidden_dims, repr_dims, eval_protocols)

    failures = FailureLog(rank)
    total = len(all_datasets)
    for iteration, seed in enumerate(default_seeds(args.iterations), start=1):
        skip = len(completed(seed) & set(all_datasets))
        main_print(f"\n>>> ITERATION {iteration}/{args.iterations} | Seed: {seed}")
        main_print(f"  Total: {total}, Skip: {skip}, Remaining: {total - skip}")

        for idx, dataset in iter_work(all_datasets, rank, world_size, f'ckpt_{args.method}'):
            if dataset in completed(seed):
                continue
            ok = failures.run(
                f'{dataset} (seed {seed})', run_classification_checkpoints,
                method=args.method, dataset=dataset, checkpoint_epochs=checkpoint_epochs,
                device=device, data_type=data_type_map[dataset], seed=seed,
                eval_protocols=eval_protocols, eval_csv=args.eval_csv, **method_kwargs,
            )
            if ok:
                print(f"\n[GPU {rank}] [{idx + 1}/{total}] Done: {dataset} (seed {seed})\n")

        main_print(f"\n{'='*70}")
        main_print(f"{args.method.upper()} CHECKPOINT ITERATION {iteration} COMPLETE")
        main_print(f"{'='*70}")

    failures.finish()


if __name__ == '__main__':
    main()
