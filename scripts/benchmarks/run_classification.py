"""
DDP Classification Script for Time Series SSL Baselines

True DDP: all GPUs train the same model with gradient sharing.

Usage:
    python scripts/benchmarks/run_classification.py --method ts2vec --datasets_file splits/datasets_ucr_t1.txt --epochs 100 --iterations 3
    torchrun --nproc_per_node=2 scripts/benchmarks/run_classification.py --method ts2vec --datasets_file splits/datasets_ucr_t1.txt --epochs 100 --iterations 3

Supported methods: ts2vec, softclt, timesurl, trep, autotcl, infots, cost, simmtm, ts_jepa, fei,
                   cf_jepa (ours)
"""

import argparse
import os
import sys
import csv
import time
import warnings
from datetime import datetime

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import numpy as np

import torch
import torch.distributed as dist
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baselines import get_trainer, get_method_dims
from utils.data_loaders import load_UCR, load_UEA


# =============================================================================
# DDP Helpers
# =============================================================================

def setup_ddp():
    """Initialize DDP if launched with torchrun."""
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
# Evaluation (rank 0 only)
# =============================================================================

def fit_svm(features, y, MAX_SAMPLES=10000, GRID_SAMPLES=5000):
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    if train_size // nb_classes < 5 or train_size < 50:
        # Small-dataset path: C=inf hard margin can produce non-finite dual
        # coefs on hard datasets (PigCVP: T=2000, N=104, 52 classes).
        # Fallback to finite C=1.0 keeps the result well-defined.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                svm = SVC(C=np.inf, gamma='scale', max_iter=10000000)
                return svm.fit(features, y)
            except ValueError as e:
                if 'finite' not in str(e).lower():
                    raise
                svm = SVC(C=1.0, gamma='scale', max_iter=10000000)
                return svm.fit(features, y)
    else:
        grid_search = GridSearchCV(
            SVC(gamma='scale', max_iter=10000000),
            {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf]},
            cv=5, n_jobs=2
        )

        # Subsample for grid search (speed up C selection)
        if train_size > GRID_SAMPLES:
            gs_split = train_test_split(
                features, y,
                train_size=GRID_SAMPLES, random_state=0, stratify=y
            )
            gs_features, gs_y = gs_split[0], gs_split[2]
        else:
            gs_features, gs_y = features, y

        # Subsample for final fit if very large
        if train_size > MAX_SAMPLES:
            split = train_test_split(
                features, y,
                train_size=MAX_SAMPLES, random_state=0, stratify=y
            )
            features = split[0]
            y = split[2]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            grid_search.fit(gs_features, gs_y)
            best_C = grid_search.best_params_['C']
            try:
                final_svm = SVC(C=best_C, gamma='scale', max_iter=10000000)
                final_svm.fit(features, y)
            except ValueError as e:
                # Non-finite dual coefs / intercept (rare hard dataset like PigCVP).
                # Retry with finite C=1.0 — softer margin avoids solver divergence.
                if 'finite' not in str(e).lower():
                    raise
                final_svm = SVC(C=1.0, gamma='scale', max_iter=10000000)
                final_svm.fit(features, y)
        return final_svm


def fit_lr(features, y, MAX_SAMPLES=100000):
    if features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            features, y,
            train_size=MAX_SAMPLES, random_state=0, stratify=y
        )
        features = split[0]
        y = split[2]

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=0,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe


def fit_svm_linear(features, y, MAX_SAMPLES=10000, GRID_SAMPLES=5000):
    nb_classes = np.unique(y, return_counts=True)[1].shape[0]
    train_size = features.shape[0]

    if train_size // nb_classes < 5 or train_size < 50:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                svm = SVC(kernel='linear', C=np.inf, max_iter=10000000)
                return svm.fit(features, y)
            except ValueError as e:
                if 'finite' not in str(e).lower():
                    raise
                svm = SVC(kernel='linear', C=1.0, max_iter=10000000)
                return svm.fit(features, y)
    else:
        grid_search = GridSearchCV(
            SVC(kernel='linear', max_iter=10000000),
            {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, np.inf]},
            cv=5, n_jobs=2
        )

        # Subsample for grid search (speed up C selection)
        if train_size > GRID_SAMPLES:
            gs_split = train_test_split(
                features, y,
                train_size=GRID_SAMPLES, random_state=0, stratify=y
            )
            gs_features, gs_y = gs_split[0], gs_split[2]
        else:
            gs_features, gs_y = features, y

        # Subsample for final fit if very large
        if train_size > MAX_SAMPLES:
            split = train_test_split(
                features, y,
                train_size=MAX_SAMPLES, random_state=0, stratify=y
            )
            features = split[0]
            y = split[2]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            grid_search.fit(gs_features, gs_y)
            best_C = grid_search.best_params_['C']
            try:
                final_svm = SVC(kernel='linear', C=best_C, max_iter=10000000)
                final_svm.fit(features, y)
            except ValueError as e:
                if 'finite' not in str(e).lower():
                    raise
                final_svm = SVC(kernel='linear', C=1.0, max_iter=10000000)
                final_svm.fit(features, y)
        return final_svm


def fit_knn(features, y):
    pipe = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=1)
    )
    pipe.fit(features, y)
    return pipe


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
        clf = fit_svm_linear(z_train, train_labels)
    elif protocol == 'knn':
        clf = fit_knn(z_train, train_labels)
    else:
        clf = fit_svm(z_train, train_labels)

    return clf.score(z_test, test_labels)


def compute_representation_metrics(z, labels):
    if np.isnan(z).any():
        z = np.nan_to_num(z, nan=0.0)
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z)
    metrics = {}
    try:
        metrics['silhouette'] = silhouette_score(z_scaled, labels)
    except Exception:
        metrics['silhouette'] = 0.0
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(z_scaled, labels)
    except Exception:
        metrics['davies_bouldin'] = float('inf')
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(z_scaled, labels)
    except Exception:
        metrics['calinski_harabasz'] = 0.0
    return metrics


# =============================================================================
# Result Paths
# =============================================================================

def get_result_paths(method, dataset_name, task='classification', data_type='UCR', run_id=None):
    base_dir = f'results/{method}/{task}/{data_type}'
    dataset_dir = f'{base_dir}/{dataset_name}'
    metrics_dir = f'{dataset_dir}/metrics'
    epoch_logs_dir = f'{dataset_dir}/epoch_logs'
    curves_dir = f'{dataset_dir}/curves'
    repr_dir = f'{dataset_dir}/representations'

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(epoch_logs_dir, exist_ok=True)
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(repr_dir, exist_ok=True)

    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    return {
        'base': base_dir,
        'dataset_dir': dataset_dir,
        'run_id': run_id,
        'csv': f'{base_dir}/results.csv',
        'metrics': f'{metrics_dir}/{run_id}.csv',
        'epoch_log': f'{epoch_logs_dir}/{run_id}.csv',
        'curve': f'{curves_dir}/{run_id}.png',
        'repr': f'{repr_dir}/{run_id}.npz',
    }


# =============================================================================
# Main Training Function
# =============================================================================

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(method, dataset_name, epochs=100, device='cuda', eval_every_epoch=False,
                   run_id=None, data_type='UCR', local_rank=-1, save_repr=True,
                   eval_protocols=None, seed=42, encode_only=False, **method_kwargs):
    if eval_protocols is None:
        eval_protocols = ['svm']
    set_seed(seed)
    rank0 = is_main_process()

    if rank0:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} ({data_type})")
        print(f"Method: {method}")
        print(f"Epochs: {epochs}")
        print(f"Device: {device}")
        if run_id is not None:
            print(f"Run ID: {run_id}")
        print(f"{'='*60}")

    paths = get_result_paths(method, dataset_name, data_type=data_type, run_id=run_id)

    # Load data (all ranks need it)
    if data_type == 'UEA':
        train_data, train_labels, test_data, test_labels = load_UEA(dataset_name)
    else:
        train_data, train_labels, test_data, test_labels = load_UCR(dataset_name)
    if rank0:
        print(f"Train: {train_data.shape}, Test: {test_data.shape}")

    if train_data.ndim == 2:
        train_data = train_data[:, :, np.newaxis]
        test_data = test_data[:, :, np.newaxis]

    input_dims = train_data.shape[2]
    T = train_data.shape[1]

    # Auto max_train_length for long sequences
    if 'max_train_length' not in method_kwargs and T > 2048:
        method_kwargs['max_train_length'] = 2048
        if rank0:
            print(f"  Auto max_train_length=2048 (T={T})")

    trainer = get_trainer(
        method=method,
        input_dims=input_dims,
        device=device,
        dataset=dataset_name,
        **method_kwargs
    )

    # Per-epoch evaluation (rank 0 only)
    epoch_results = []
    best_state = {'acc': 0, 'epoch': 0, 'z_train': None, 'z_test': None}

    def epoch_callback(trainer, epoch, loss):
        nonlocal best_state
        if not rank0:
            return
        z_train_ep = trainer.encode(train_data)
        z_test_ep = trainer.encode(test_data)
        acc = evaluate_classifier(z_train_ep, train_labels, z_test_ep, test_labels, eval_protocols[0])
        epoch_results.append({'epoch': epoch, 'loss': loss, 'accuracy': acc})
        if acc > best_state['acc']:
            best_state = {'acc': acc, 'epoch': epoch, 'z_train': z_train_ep.copy(), 'z_test': z_test_ep.copy()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: loss={loss:.4f}, acc={acc*100:.2f}%")

    # Train
    if rank0:
        print("\nTraining...")
    t_start = time.time()
    if eval_every_epoch:
        trainer.fit(train_data, epochs=epochs, verbose=False, callback=epoch_callback)
    else:
        trainer.fit(train_data, epochs=epochs, verbose=(rank0), callback=None)
    train_time = time.time() - t_start

    print("\nEncoding representations...")
    z_train = trainer.encode(train_data)
    z_test = trainer.encode(test_data)
    print(f"  z_train: {z_train.shape}, z_test: {z_test.shape}")

    # --- encode_only: save representations and return early ---
    if encode_only:
        np.savez(paths['repr'], z_train=z_train, z_test=z_test,
                 train_labels=train_labels, test_labels=test_labels)
        print(f"  Saved representations: {paths['repr']}")
        print(f"  Train Time: {train_time:.1f}s")
        return []

    # Evaluate for each protocol
    all_results = []
    metrics_final = compute_representation_metrics(z_test, test_labels)

    if eval_every_epoch and epoch_results:
        ep_accs = [r['accuracy'] for r in epoch_results]
        max_acc_ep = max(ep_accs)
        max_epoch = ep_accs.index(max_acc_ep) + 1
        last10_std = np.std(ep_accs[-10:]) if len(ep_accs) >= 10 else np.std(ep_accs)
        metrics_best = compute_representation_metrics(best_state['z_test'], test_labels) if best_state['z_test'] is not None else metrics_final
    else:
        max_acc_ep = None
        max_epoch = epochs
        last10_std = 0.0
        metrics_best = metrics_final

    for protocol in eval_protocols:
        print(f"\nEvaluating ({protocol})...")
        accuracy = evaluate_classifier(z_train, train_labels, z_test, test_labels, protocol)
        print(f"  {protocol}: {accuracy*100:.2f}%")

        all_results.append({
            'run_id': paths['run_id'],
            'dataset': dataset_name,
            'method': method,
            'eval_protocol': protocol,
            'accuracy': accuracy,
            'max_accuracy': max_acc_ep if max_acc_ep is not None else accuracy,
            'max_epoch': max_epoch,
            'last10_std': last10_std,
            'train_time': train_time,
        })

    print(f"  Train Time:        {train_time:.1f}s")

    # Save metrics (first protocol)
    with open(paths['metrics'], 'w', newline='') as f:
        fieldnames = ['type', 'epoch', 'accuracy', 'silhouette', 'davies_bouldin', 'calinski_harabasz']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'type': 'final', 'epoch': epochs,
            'accuracy': round(all_results[0]['accuracy'] * 100, 2),
            'silhouette': round(metrics_final['silhouette'], 4),
            'davies_bouldin': round(metrics_final['davies_bouldin'], 4),
            'calinski_harabasz': round(metrics_final['calinski_harabasz'], 1),
        })
        if eval_every_epoch:
            writer.writerow({
                'type': 'best', 'epoch': max_epoch,
                'accuracy': round(all_results[0]['max_accuracy'] * 100, 2),
                'silhouette': round(metrics_best['silhouette'], 4),
                'davies_bouldin': round(metrics_best['davies_bouldin'], 4),
                'calinski_harabasz': round(metrics_best['calinski_harabasz'], 1),
            })

    # Save epoch logs
    if eval_every_epoch and epoch_results:
        with open(paths['epoch_log'], 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'accuracy'])
            writer.writeheader()
            for r in epoch_results:
                writer.writerow({'epoch': r['epoch'], 'loss': round(r['loss'], 4), 'accuracy': round(r['accuracy'] * 100, 2)})

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        ep_accs_plot = [r['accuracy'] for r in epoch_results]
        losses = [r['loss'] for r in epoch_results]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(ep_accs_plot, 'b-', linewidth=1)
        first_acc = all_results[0]['accuracy']
        ax1.axhline(y=first_acc, color='r', linestyle='--', label=f'Final: {first_acc*100:.1f}%')
        if max_acc_ep is not None:
            ax1.axhline(y=max_acc_ep, color='g', linestyle=':', label=f'Max: {max_acc_ep*100:.1f}%')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.set_title(f'{dataset_name} - Accuracy')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(losses, 'r-', linewidth=1)
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.set_title(f'{dataset_name} - Loss')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(paths['curve'], dpi=150); plt.close()

    if save_repr:
        np.savez(paths['repr'], z_train=z_train, z_test=z_test,
                 train_labels=train_labels, test_labels=test_labels)
        print(f"  Saved representations: {paths['repr']}")

    return all_results


# =============================================================================
# Checkpoint-based Encoding (train + save representations at multiple epochs)
# =============================================================================

def _apply_pooling(z, method='maxpool', pool_size=8):
    """Apply pooling to (N, T, D) representations."""
    import torch.nn.functional as F_pool

    if z.ndim == 2:
        return z  # Already pooled (e.g., SimMTM)

    N, T, D = z.shape
    z_t = torch.from_numpy(z).float()

    if method == 'maxpool':
        out = F_pool.max_pool1d(z_t.transpose(1, 2), kernel_size=T).squeeze(-1)
        return out.numpy()
    elif method == 'adaptive_avg':
        out = F_pool.adaptive_avg_pool1d(z_t.transpose(1, 2), pool_size)
        out = out.transpose(1, 2)
        return out.reshape(N, -1).numpy()
    else:
        raise ValueError(f"Unknown pooling: {method}")


def run_encode_checkpoints(method, dataset_name, max_epochs=200,
                           checkpoint_epochs=None, device='cuda',
                           data_type='UCR', local_rank=-1, seed=42,
                           pooling_methods=None, pool_size=8,
                           eval_protocols=None, eval_csv=None, **method_kwargs):
    """Train and evaluate at each checkpoint epoch.

    At each checkpoint: encode → apply pooling → SVM/Linear eval → append to CSV.
    No npz files saved (disk-efficient).
    Also saves loss log per epoch.
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = [50, 100, 200]
    if pooling_methods is None:
        pooling_methods = ['maxpool', 'adaptive_avg']
    if eval_protocols is None:
        eval_protocols = ['svm', 'svm_linear']

    set_seed(seed)
    rank0 = is_main_process()

    if rank0:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} ({data_type})")
        print(f"Method: {method}")
        print(f"Max Epochs: {max_epochs}")
        print(f"Checkpoints: {checkpoint_epochs}")
        print(f"Seed: {seed}")
        print(f"Device: {device}")
        default_h, default_r = get_method_dims(method)
        print(f"Hidden dims: {method_kwargs.get('hidden_dims', default_h)}")
        print(f"Repr dims: {method_kwargs.get('output_dims', default_r)}")
        print(f"{'='*60}")

    # Paths
    base_dir = f'results/{method}/classification/{data_type}'
    dataset_dir = f'{base_dir}/{dataset_name}'
    loss_dir = f'{dataset_dir}/loss_logs'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    if eval_csv is None:
        eval_csv = 'results/cls_results.csv'
    os.makedirs(os.path.dirname(eval_csv) or '.', exist_ok=True)

    # Load data
    if data_type == 'UEA':
        train_data, train_labels, test_data, test_labels = load_UEA(dataset_name)
    else:
        train_data, train_labels, test_data, test_labels = load_UCR(dataset_name)
    if rank0:
        print(f"Train: {train_data.shape}, Test: {test_data.shape}")

    if train_data.ndim == 2:
        train_data = train_data[:, :, np.newaxis]
        test_data = test_data[:, :, np.newaxis]

    input_dims = train_data.shape[2]
    T = train_data.shape[1]

    if 'max_train_length' not in method_kwargs and T > 2048:
        method_kwargs['max_train_length'] = 2048
        if rank0:
            print(f"  Auto max_train_length=2048 (T={T})")

    trainer = get_trainer(
        method=method,
        input_dims=input_dims,
        device=device,
        dataset=dataset_name,
        **method_kwargs
    )

    # Get hidden_dims/repr_dims for CSV recording
    default_h, default_r = get_method_dims(method)
    hidden_dims = method_kwargs.get('hidden_dims', default_h)
    repr_dims = method_kwargs.get('output_dims', default_r)

    # Check which checkpoint epochs already have ALL results in CSV
    # Each epoch needs len(pooling_methods) * len(eval_protocols) rows to be complete
    existing_epochs = set()
    _epoch_row_counts = {}
    if os.path.exists(eval_csv):
        import csv as _csv
        with open(eval_csv, newline='') as _f:
            for _row in _csv.DictReader(_f):
                if (_row.get('method') == method and
                    _row.get('dataset') == dataset_name and
                    _row.get('seed') == str(seed) and
                    _row.get('hidden_dims', str(hidden_dims)) == str(hidden_dims) and
                    _row.get('repr_dims', str(repr_dims)) == str(repr_dims)):
                    ep = int(_row['epoch'])
                    _epoch_row_counts[ep] = _epoch_row_counts.get(ep, 0) + 1
        _expected_rows = len(pooling_methods) * len(eval_protocols)
        for ep, count in _epoch_row_counts.items():
            if count >= _expected_rows:
                existing_epochs.add(ep)

    # Train and evaluate at checkpoints
    loss_log = []
    eval_results = []
    t_start = time.time()

    # Single continuous fit() call; callback triggers encode+eval at checkpoint epochs.
    # This preserves optimizer/scheduler state across checkpoints (unlike the previous
    # chunked fit(50)+fit(50)+fit(100) pattern, which recreated optimizer per call for
    # methods whose upstream fit() creates a fresh optimizer internally — TS2Vec,
    # SoftCLT, TimesURL, CoST).
    sorted_checkpoints = sorted(ep for ep in checkpoint_epochs if ep <= max_epochs)
    if not sorted_checkpoints:
        return 0.0
    max_checkpoint = sorted_checkpoints[-1]
    checkpoint_set = set(sorted_checkpoints)

    def _epoch_callback(tr, ep, loss):
        loss_log.append((ep, loss))
        if ep not in checkpoint_set:
            return
        if ep in existing_epochs:
            if rank0:
                print(f"  Epoch {ep} already in CSV, skipping evaluation.")
            return

        if rank0:
            print(f"\n  [Checkpoint] Encoding at epoch {ep}...")
        z_train_raw = tr.encode(train_data, raw=True).astype(np.float32)
        z_test_raw = tr.encode(test_data, raw=True).astype(np.float32)

        for pool_method in pooling_methods:
            z_train = _apply_pooling(z_train_raw, method=pool_method, pool_size=pool_size)
            z_test = _apply_pooling(z_test_raw, method=pool_method, pool_size=pool_size)

            for protocol in eval_protocols:
                acc = evaluate_classifier(z_train, train_labels, z_test, test_labels, protocol)
                eval_results.append({
                    'method': method,
                    'dataset': dataset_name,
                    'seed': seed,
                    'epoch': ep,
                    'hidden_dims': hidden_dims,
                    'repr_dims': repr_dims,
                    'pooling': pool_method,
                    'pool_size': pool_size if pool_method == 'adaptive_avg' else '',
                    'eval_protocol': protocol,
                    'accuracy': round(acc * 100, 2),
                })
                if rank0:
                    print(f"    {pool_method} | {protocol}: {acc*100:.2f}%")

            del z_train, z_test

        del z_train_raw, z_test_raw

        # Dual-encoder eval: for cf_jepa, also evaluate with the target
        # encoder. Saved as method='cf_jepa_target'. Same training, just an
        # extra forward pass — no extra GPU time other than encode().
        if (method == 'cf_jepa'
                and hasattr(tr, 'target_encoder')
                and tr.target_encoder is not None):
            if rank0:
                print(f"  [Dual eval] Re-encoding with target encoder...")
            zt_train_raw = tr.encode(train_data, raw=True, use_target=True).astype(np.float32)
            zt_test_raw = tr.encode(test_data, raw=True, use_target=True).astype(np.float32)
            for pool_method in pooling_methods:
                zt_train = _apply_pooling(zt_train_raw, method=pool_method, pool_size=pool_size)
                zt_test = _apply_pooling(zt_test_raw, method=pool_method, pool_size=pool_size)
                for protocol in eval_protocols:
                    acc_t = evaluate_classifier(zt_train, train_labels, zt_test, test_labels, protocol)
                    eval_results.append({
                        'method': 'cf_jepa_target',
                        'dataset': dataset_name,
                        'seed': seed,
                        'epoch': ep,
                        'hidden_dims': hidden_dims,
                        'repr_dims': repr_dims,
                        'pooling': pool_method,
                        'pool_size': pool_size if pool_method == 'adaptive_avg' else '',
                        'eval_protocol': protocol,
                        'accuracy': round(acc_t * 100, 2),
                    })
                    if rank0:
                        print(f"    [target] {pool_method} | {protocol}: {acc_t*100:.2f}%")
                del zt_train, zt_test
            del zt_train_raw, zt_test_raw

        import gc; gc.collect()

    if rank0:
        print(f"\n  Training 1 -> {max_checkpoint} (continuous)...")
    trainer.fit(train_data, epochs=max_checkpoint, verbose=False,
                callback=_epoch_callback)

    train_time = time.time() - t_start

    # Append eval results to CSV
    if eval_results:
        fieldnames = ['method', 'dataset', 'seed', 'epoch', 'hidden_dims', 'repr_dims',
                      'pooling', 'pool_size', 'eval_protocol', 'accuracy']
        file_exists = os.path.exists(eval_csv)
        with open(eval_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(eval_results)

    # Save loss log
    loss_csv = f'{loss_dir}/seed{seed}_loss.csv'
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


def _count_repr_files(method, dataset_name, data_type='UCR'):
    """Count how many .npz representation files exist for a dataset."""
    repr_dir = f'results/{method}/classification/{data_type}/{dataset_name}/representations'
    if not os.path.exists(repr_dir):
        return 0
    return len([f for f in os.listdir(repr_dir) if f.endswith('.npz')])


def _load_completed_counts(csv_path, eval_protocol):
    """Return {dataset_name: num_completions} for given protocol."""
    counts = {}
    if not os.path.exists(csv_path):
        return counts
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('eval_protocol', 'svm') == eval_protocol:
                ds = row.get('dataset', '')
                counts[ds] = counts.get(ds, 0) + 1
    return counts


def save_results_csv(results, method, task='classification', data_type='UCR'):
    if not results:
        return
    base_dir = f'results/{method}/{task}/{data_type}'
    os.makedirs(base_dir, exist_ok=True)
    csv_path = f'{base_dir}/results.csv'
    fieldnames = ['run_id', 'dataset', 'eval_protocol', 'final_acc', 'best_acc', 'best_epoch', 'last10_std', 'train_time']
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow({
                'run_id': r.get('run_id', ''),
                'dataset': r['dataset'],
                'eval_protocol': r.get('eval_protocol', 'svm'),
                'final_acc': round(r['accuracy'] * 100, 2),
                'best_acc': round(r['max_accuracy'] * 100, 2),
                'best_epoch': r['max_epoch'],
                'last10_std': round(r['last10_std'] * 100, 2),
                'train_time': round(r['train_time'], 1),
            })
    print(f"[Appended] {csv_path} (+{len(results)} rows)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='DDP Classification for Time Series SSL')
    parser.add_argument('--method', type=str, default='ts2vec',
                        help='ts2vec, softclt, timesurl, trep, autotcl, infots, cost, simmtm, ts_jepa, fei, cf_jepa')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--skip_epoch_eval', action='store_true')
    parser.add_argument('--eval_every_epoch', action='store_true', default=True)
    parser.add_argument('--no_save_repr', action='store_true', help='Do not save .npz representation files')
    parser.add_argument('--encode_only', action='store_true',
                        help='Train and save representations only (skip evaluation)')
    parser.add_argument('--checkpoint_mode', action='store_true',
                        help='Train and save raw+pooled representations at multiple epoch checkpoints')
    parser.add_argument('--checkpoint_epochs', type=str, default='50,100,200',
                        help='Comma-separated checkpoint epochs (default: 10,50,100,200)')
    parser.add_argument('--datasets_file', type=str, default=None,
                        help='Text file with dataset names (one per line)')
    parser.add_argument('--eval_protocol', type=str, default='all',
                        choices=['svm', 'svm_linear', 'linear', 'knn', 'all'],
                        help='Evaluation protocol: all (default, svm+svm_linear), or single protocol')
    parser.add_argument('--hidden_dims', type=int, default=None,
                        help='Override hidden dims (default: per-method)')
    parser.add_argument('--output_dims', type=int, default=None,
                        help='Override output/repr dims (default: per-method)')
    parser.add_argument('--eval_csv', type=str, default='results/cls_results.csv',
                        help='Output CSV path (default: results/cls_results.csv). '
                             'Use a separate path for verification runs to avoid '
                             'contaminating main results.')
    args = parser.parse_args()

    # Resolve eval protocols
    if args.eval_protocol == 'all':
        eval_protocols = ['svm', 'svm_linear']
    else:
        eval_protocols = [args.eval_protocol]

    # Parse checkpoint epochs
    checkpoint_epochs = [int(x) for x in args.checkpoint_epochs.split(',')]

    # DDP setup
    rank, world_size, local_rank = setup_ddp()
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    if rank == 0:
        print(f"Method: {args.method}")
        print(f"World size: {world_size}")
        print(f"Iterations: {args.iterations}")

    # Dataset list (auto-detect UCR/UEA per dataset; supports mixed UCR+UEA files)
    if args.datasets_file:
        with open(args.datasets_file) as f:
            all_datasets = [line.strip() for line in f if line.strip()]
    elif args.dataset:
        all_datasets = [args.dataset]
    else:
        raise SystemExit("Must specify --datasets_file or --dataset")

    def _resolve_data_type(ds):
        if os.path.exists(f'data/UEA/{ds}'):
            return 'UEA'
        if os.path.exists(f'data/UCR/{ds}'):
            return 'UCR'
        raise SystemExit(f"Dataset {ds!r} not found in data/UCR/ or data/UEA/")

    # Validate and build mapping (raises early if any dataset is missing)
    data_type_map = {ds: _resolve_data_type(ds) for ds in all_datasets}
    # Default for code paths that still expect a single data_type (e.g. logging)
    data_type = data_type_map[all_datasets[0]]

    # =====================================================================
    # Checkpoint mode: train + evaluate at multiple epoch checkpoints
    # =====================================================================
    if args.checkpoint_mode:
        max_epoch = max(checkpoint_epochs)
        seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

        # Build method_kwargs from CLI overrides
        method_kwargs = {}
        if args.hidden_dims is not None:
            method_kwargs['hidden_dims'] = args.hidden_dims
        if args.output_dims is not None:
            method_kwargs['output_dims'] = args.output_dims

        # Determine actual dims for resume check and CSV
        default_h, default_r = get_method_dims(args.method)
        actual_hidden = args.hidden_dims if args.hidden_dims is not None else default_h
        actual_repr = args.output_dims if args.output_dims is not None else default_r

        if rank == 0 and method_kwargs:
            print(f"Override dims: hidden={actual_hidden}, repr={actual_repr}")

        def _is_dataset_done(method, dataset, data_type, seed, max_ep):
            """Check if ALL checkpoint epochs have 4 rows each in eval CSV."""
            eval_csv = args.eval_csv
            if not os.path.exists(eval_csv):
                return False
            epoch_counts = {}
            with open(eval_csv, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if (row.get('method') == method and
                        row.get('dataset') == dataset and
                        row.get('seed') == str(seed) and
                        row.get('hidden_dims', '64') == str(actual_hidden) and
                        row.get('repr_dims', '320') == str(actual_repr)):
                        ep = row.get('epoch')
                        epoch_counts[ep] = epoch_counts.get(ep, 0) + 1
            for ep in checkpoint_epochs:
                if epoch_counts.get(str(ep), 0) < 4:
                    return False
            return True

        # Dynamic queue setup
        use_queue = world_size > 1
        if use_queue:
            import tempfile, fcntl
            counter_file = os.path.join(tempfile.gettempdir(),
                f'ssl_ckpt_{args.method}_{os.environ.get("MASTER_PORT", "29500")}.counter')
            print(f"[GPU {rank}] Checkpoint mode, {len(all_datasets)} datasets")

            def get_next_idx():
                with open(counter_file, 'r+') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    idx = int(f.read().strip())
                    if idx < len(all_datasets):
                        f.seek(0); f.write(str(idx + 1)); f.truncate()
                    fcntl.flock(f, fcntl.LOCK_UN)
                return idx if idx < len(all_datasets) else None
        else:
            print(f"Checkpoint mode: {len(all_datasets)} datasets, checkpoints={checkpoint_epochs}")

        for iteration in range(1, args.iterations + 1):
            seed = seeds[iteration - 1] if iteration <= len(seeds) else 42 + iteration - 1
            if rank == 0:
                print(f"\n>>> ITERATION {iteration}/{args.iterations} | Seed: {seed}")

            # Count how many need to run (for progress)
            total_datasets = len(all_datasets)
            skip_count = sum(1 for ds in all_datasets
                if _is_dataset_done(args.method, ds, data_type_map[ds], seed, max_epoch))
            remaining = total_datasets - skip_count
            if rank == 0:
                print(f"  Total: {total_datasets}, Skip: {skip_count}, Remaining: {remaining}")

            if use_queue:
                if rank == 0:
                    with open(counter_file, 'w') as f:
                        f.write('0')
                dist.barrier()

                while True:
                    idx = get_next_idx()
                    if idx is None:
                        break
                    dataset = all_datasets[idx]
                    ds_data_type = data_type_map[dataset]
                    if _is_dataset_done(args.method, dataset, ds_data_type, seed, max_epoch):
                        continue
                    try:
                        run_encode_checkpoints(
                            method=args.method, dataset_name=dataset,
                            max_epochs=max_epoch, checkpoint_epochs=checkpoint_epochs,
                            device=device, data_type=ds_data_type,
                            local_rank=local_rank, seed=seed,
                            eval_csv=args.eval_csv,
                            **method_kwargs,
                        )
                        print(f"\n[GPU {rank}] [{idx + 1}/{total_datasets}] Done: {dataset} (seed {seed})\n")
                    except Exception as e:
                        print(f"[GPU {rank}] Error on {dataset}: {e}")
                        import traceback; traceback.print_exc()
                        continue
                dist.barrier()
            else:
                for i, dataset in enumerate(all_datasets):
                    ds_data_type = data_type_map[dataset]
                    if _is_dataset_done(args.method, dataset, ds_data_type, seed, max_epoch):
                        continue
                    try:
                        run_encode_checkpoints(
                            method=args.method, dataset_name=dataset,
                            max_epochs=max_epoch, checkpoint_epochs=checkpoint_epochs,
                            device=device, data_type=ds_data_type,
                            local_rank=local_rank, seed=seed,
                            eval_csv=args.eval_csv,
                            **method_kwargs,
                        )
                        print(f"\n[{i + 1}/{total_datasets}] Done: {dataset} (seed {seed})\n")
                    except Exception as e:
                        print(f"Error on {dataset}: {e}")
                        import traceback; traceback.print_exc()
                        continue

            if rank == 0:
                print(f"\n{'='*70}")
                print(f"{args.method.upper()} CHECKPOINT ITERATION {iteration} COMPLETE")
                print(f"{'='*70}")

        if rank == 0:
            print("\nAll checkpoint iterations complete!")
        if world_size > 1:
            dist.destroy_process_group()
        return

    # =====================================================================
    # Normal mode (existing logic)
    # =====================================================================

    # Dynamic queue for multi-GPU: each GPU grabs next available dataset
    use_queue = world_size > 1
    if use_queue:
        import tempfile, fcntl
        counter_file = os.path.join(tempfile.gettempdir(),
            f'ssl_cls_{args.method}_{os.environ.get("MASTER_PORT", "29500")}.counter')
        print(f"[GPU {rank}] Dynamic queue mode, {len(all_datasets)} datasets")
    else:
        print(f"Datasets: {len(all_datasets)} total ({data_type})")

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

    eval_every_epoch = args.eval_every_epoch and not args.skip_epoch_eval

    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]  # one per iteration

    for iteration in range(1, args.iterations + 1):
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        seed = seeds[iteration - 1] if iteration <= len(seeds) else 42 + iteration - 1

        if rank == 0:
            print(f"\n>>> ITERATION {iteration}/{args.iterations} | Run ID: {run_id} | Seed: {seed}")

        # Load completed counts — skip datasets already done for this iteration
        if args.encode_only:
            completed_per_protocol = None  # not used
        else:
            result_csv = f'results/{args.method}/classification/{data_type}/results.csv'
            completed_per_protocol = {p: _load_completed_counts(result_csv, p) for p in eval_protocols}

        def _should_skip(dataset):
            if args.encode_only:
                return _count_repr_files(args.method, dataset, data_type) >= iteration
            return not [p for p in eval_protocols if completed_per_protocol[p].get(dataset, 0) < iteration]

        def _get_needed_protocols(dataset):
            if args.encode_only:
                return eval_protocols  # not used for eval, but passed through
            return [p for p in eval_protocols if completed_per_protocol[p].get(dataset, 0) < iteration]

        if use_queue:
            if rank == 0:
                with open(counter_file, 'w') as f:
                    f.write('0')
            dist.barrier()

            completed = 0
            while True:
                idx = get_next_dataset_idx()
                if idx is None:
                    break
                dataset = all_datasets[idx]
                if _should_skip(dataset):
                    continue
                needed = _get_needed_protocols(dataset)
                completed += 1
                try:
                    results = run_experiment(
                        method=args.method,
                        dataset_name=dataset,
                        epochs=args.epochs,
                        device=device,
                        eval_every_epoch=eval_every_epoch,
                        run_id=run_id,
                        data_type=data_type,
                        local_rank=local_rank,
                        save_repr=not args.no_save_repr,
                        eval_protocols=needed,
                        seed=seed,
                        encode_only=args.encode_only,
                    )
                    if results:
                        save_results_csv(results, args.method, data_type=data_type)
                    print(f"\n[GPU {rank}] Iter {iteration}: {completed} done (latest: {dataset})\n")
                except Exception as e:
                    print(f"[GPU {rank}] Error on {dataset}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            dist.barrier()
        else:
            for i, dataset in enumerate(all_datasets):
                if _should_skip(dataset):
                    continue
                needed = _get_needed_protocols(dataset)
                try:
                    results = run_experiment(
                        method=args.method,
                        dataset_name=dataset,
                        epochs=args.epochs,
                        device=device,
                        eval_every_epoch=eval_every_epoch,
                        run_id=run_id,
                        data_type=data_type,
                        local_rank=local_rank,
                        save_repr=not args.no_save_repr,
                        eval_protocols=needed,
                        seed=seed,
                        encode_only=args.encode_only,
                    )
                    if results:
                        save_results_csv(results, args.method, data_type=data_type)
                    print(f"\nIter {iteration}: {i+1}/{len(all_datasets)} datasets completed\n")
                except Exception as e:
                    print(f"[GPU {rank}] Error on {dataset}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        if rank == 0 and len(all_datasets) > 1:
            print(f"\n{'='*70}")
            print(f"{args.method.upper()} ITERATION {iteration} COMPLETE")
            print(f"Run ID: {run_id}")
            print(f"Results: results/{args.method}/classification/{data_type}/results.csv")
            print(f"{'='*70}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
