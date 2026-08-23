"""SoftCLT trainer wrapper.

Fair comparison: uses fixed hyperparameters (tau_inst=5.0, tau_temp=1.0)
across all datasets — the median of the paper's per-dataset tuned values
(see softclt_ts2vec/hyperparameters/cls_hyperparams.csv). This activates
both soft-CL mechanisms (instance + temporal) without per-dataset tuning,
matching the paper's algorithmic contribution under a fair protocol.
"""

import os
import sys
import warnings
import shutil
import tempfile
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
from baselines._import_helper import isolated_import

_REPO_DIR = os.path.join(os.path.dirname(__file__), "softclt", "softclt_ts2vec")

# SoftCLT imports tensorboard at module level; mock it if not installed.
try:
    import tensorboard  # noqa: F401
except ImportError:
    import types
    _tb_mod = types.ModuleType('tensorboard')
    sys.modules['tensorboard'] = _tb_mod
    _sw_mod = types.ModuleType('torch.utils.tensorboard')
    _sw_mod.SummaryWriter = type('SummaryWriter', (), {'__init__': lambda *a, **k: None})
    sys.modules['torch.utils.tensorboard'] = _sw_mod

_mod = isolated_import(_REPO_DIR, "soft_ts2vec")
TS2Vec = _mod.TS2Vec


class SoftCLTTrainer:

    def __init__(self, input_dims, output_dims=320, hidden_dims=64, depth=10,
                 device='cuda', lr=0.001, batch_size=16,
                 max_train_length=None, **kwargs):
        self._epoch_counter = 0
        self._callback = None
        self.output_dims = output_dims
        self.max_train_length = max_train_length

        # Fixed hyperparameters for fair comparison (no per-dataset tuning).
        # Values = median of paper's per-dataset CSV (n=128, tau_inst median=5.0,
        # tau_temp median=1.0). Both > 0 → both soft-CL mechanisms active.
        tau_inst = kwargs.get("tau_inst", 5.0)
        tau_temp = kwargs.get("tau_temp", 1.0)

        # Force-disable instance soft CL for anomaly datasets to mirror
        # upstream train.py:167-168 (`sim_mat=None` for forecasting / anomaly).
        # Anomaly data is (n_series, T_padded, 1) where per-series DTW between
        # different time series is structurally inapplicable and the O(N²·T²)
        # preprocessing is prohibitively slow (KPI 58×73K cells, Yahoo 367×840
        # ≈ minutes-to-days). Forecasting (N=1) is caught by the fit-time
        # check; the dataset-name guard catches anomaly's multi-series case.
        anomaly_dataset = str(kwargs.get('dataset', '')).lower() in {'kpi', 'yahoo'}
        soft_instance = (tau_inst > 0) and not anomaly_dataset
        soft_temporal = tau_temp > 0
        self._tau_inst = tau_inst
        self._soft_instance = soft_instance
        # DTW soft-label normalization axis for multivariate (N,T,C) input.
        # Default 1 (=time axis, matches upstream normalize_TS). Exposed only so
        # the axis-treatment diagnostic can reproduce the prior axis=2 (channel)
        # behavior; production always uses 1.
        self._dtw_norm_axis = kwargs.get('dtw_norm_axis', 1)

        self._model = TS2Vec(
            input_dims=input_dims, output_dims=output_dims,
            hidden_dims=hidden_dims, depth=depth, device=device,
            lr=lr, batch_size=batch_size,
            lambda_=kwargs.get("lambda_", 0.5),
            tau_temp=tau_temp,
            max_train_length=max_train_length,
            temporal_unit=kwargs.get("temporal_unit", 0),
            after_epoch_callback=self._epoch_callback_bridge,
            soft_instance=soft_instance,
            soft_temporal=soft_temporal,
        )

    def _epoch_callback_bridge(self, _softclt_self, cum_loss):
        self._epoch_counter += 1
        if self._callback is not None:
            self._callback(self, self._epoch_counter, cum_loss)

    @staticmethod
    def _compute_sim_mat(train_data, norm_axis=1):
        """Compute DTW similarity matrix (normalized to [0,1]) for instance-wise soft CL.

        Follows original SoftCLT pipeline: normalize → DTW → MinMaxScale → 1 - dist.
        norm_axis controls the multivariate min-max axis (1=time=correct/default;
        2=channel reproduces the prior wrapper bug — diagnostic only).
        """
        from tslearn.metrics import dtw
        from sklearn.preprocessing import MinMaxScaler

        # train_data: (N, T, C) — squeeze to 2D for univariate
        if train_data.shape[2] == 1:
            X = np.nan_to_num(train_data[:, :, 0], nan=0.0)
            # Normalize each sample to [0,1] (original: normalize_TS)
            xmax = X.max(axis=1, keepdims=True)
            xmin = X.min(axis=1, keepdims=True)
            X = (X - xmin) / (xmax - xmin + 1e-6)
        else:
            # Per-channel min-max over the TIME axis, matching upstream
            # normalize_TS: upstream takes MTS as (N, D=channels, L=time) and
            # reduces over axis=2 (time). Our layout is (N, T=time, C=channels),
            # so the time axis is axis=1 (NOT axis=2, which would min-max across
            # channels per timestep and corrupt the DTW instance similarity).
            X = np.nan_to_num(train_data, nan=0.0)
            xmax = X.max(axis=norm_axis, keepdims=True)
            xmin = X.min(axis=norm_axis, keepdims=True)
            X = (X - xmin) / (xmax - xmin + 1e-6)

        N = len(X)
        dist_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(i):
                if train_data.shape[2] == 1:
                    d = dtw(X[i].reshape(-1, 1), X[j].reshape(-1, 1))
                else:
                    d = dtw(X[i], X[j])
                dist_mat[i, j] = d
                dist_mat[j, i] = d

        # Normalize: set diagonal to row-min, then MinMaxScale, then 1 - dist
        diag_idx = np.diag_indices(N)
        mask = np.ones(dist_mat.shape, dtype=bool)
        mask[diag_idx] = False
        temp = dist_mat[mask].reshape(N, N - 1)
        dist_mat[diag_idx] = temp.min()

        scaler = MinMaxScaler(feature_range=(0, 1))
        dist_mat = scaler.fit_transform(dist_mat)
        sim_mat = 1 - dist_mat
        return sim_mat

    def fit(self, train_data, epochs, verbose=False, callback=None):
        self._epoch_counter = 0
        self._callback = callback
        run_dir = tempfile.mkdtemp(prefix="softclt_run_")

        # Compute soft labels for instance-wise soft CL.
        # Skip when N==1 (forecasting: single long series) or T > 5000 (anomaly:
        # full-length series, e.g. KPI T=73K / 58 series). Instance soft CL is
        # undefined for N=1 and structurally inapplicable to anomaly's
        # per-series-as-instance setup. Matches upstream train.py:167-168 which
        # forces sim_mat=None for forecasting / anomaly task types.
        # T threshold also avoids O(N²·T²) DTW preprocessing blow-up.
        N, T = train_data.shape[0], train_data.shape[1]
        soft_labels = None
        use_instance = self._soft_instance and N > 1 and T <= 5000
        if use_instance:
            # Cache sim_mat to avoid recomputation on incremental fit() calls
            if not hasattr(self, '_cached_sim_mat') or self._cached_sim_mat is None:
                self._cached_sim_mat = self._compute_sim_mat(
                    train_data, norm_axis=self._dtw_norm_axis)
            sim_mat = self._cached_sim_mat
            # densify: convert similarity → soft assignment
            alpha = 0.5
            dist_mat = 1 - sim_mat
            soft_labels = ((2 * alpha) / (1 + np.exp(-self._tau_inst * (-dist_mat)))) + \
                          (1 - alpha) * np.eye(dist_mat.shape[0])

        # Sync model.soft_instance with our actual decision: upstream hier_CL_soft
        # references soft_labels_L only inside `if soft_labels is not None`,
        # but the inst-CL branch is gated by `soft_instance` flag → UnboundLocalError
        # when soft_instance=True but soft_labels=None. Restore after fit so the
        # trainer object's invariants survive across calls.
        prev_soft_instance = self._model.soft_instance
        self._model.soft_instance = use_instance

        try:
            self._model.fit(
                train_data, train_labels=None, test_data=None,
                test_labels=None, soft_labels=soft_labels, run_dir=run_dir,
                n_epochs=self._model.n_epochs + epochs, verbose=verbose,
            )
        finally:
            self._model.soft_instance = prev_soft_instance
            shutil.rmtree(run_dir, ignore_errors=True)
        return self

    def encode(self, data, raw=False):
        if raw:
            return self._model.encode(data, encoding_window=None)
        return self._model.encode(data, encoding_window="full_series")

    @property
    def _net(self):
        return self._model._net

    @property
    def device(self):
        return self._model.device

    @property
    def model(self):
        return self._model
