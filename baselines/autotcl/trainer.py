"""AutoTCL trainer wrapper.

Unified TS2Vec backbone across cls / fcst / ano (2026-05-29 fairness unification)
------------------------------------------------------------------------------
Originally the wrapper followed paper's task-specific backbones:
  * Forecasting → CoST-no-SFD (paper Sec 4.1 main)
  * Classification → TS2Vec dilated CNN (paper Sec 4.2 main)

This gave AutoTCL a unique "best-of-both-worlds" backbone advantage no other
method in the benchmark has, breaking the single-backbone fairness convention
applied to TS2Vec / SoftCLT / TimesURL / T-Rep / InfoTS / CoST. AutoTCL's
methodological contribution is the **parametric augmentation network** (h(x),
g(x)), not the encoder choice; isolating that contribution requires a single
shared encoder family.

Resolution: TS2Vec backbone for all three tasks.
  * cls / ano keep paper-faithful config (paper Sec 4.2 "We use TS2Vec network
    architecture", paper App D.1 "we followed the experiment setting of TS2Vec").
  * fcst switches from CoST-no-SFD to TS2Vec, validated by paper App D.3.1
    Table 7 (AutoTCL with TS2Vec backbone on ETTh1/h2/m1, Elec, WTH, Lora).
    Paper Avg MSE delta: CoST 0.157 → TS2Vec 0.165 (+5%), still beating
    vanilla TS2Vec baselines per Table 7. Trade-off acknowledged.

Hyperparameters: AutoTCL_CoST.py class defaults (lr=1e-3, augmask='binomial',
hard_mask=True, aug_depth=3, gumbel_bias=0.001, gamma_zeta=0.05). These align
with paper App D.1 "TS2Vec experimental setting" reference and are applied
uniformly across cls / fcst / ano. No task-specific HP branching.
  * Replaces the prior fcst-specific median of paper's per-dataset config.py
    (lr=2e-5, augmask='mask_last', etc.). Those medians were tuned for CoST
    backbone and have no theoretical justification under TS2Vec backbone.

The TS2Vec encoder is bundled locally at ``ts2vec_backbone.py`` (verbatim
copy from the TS2Vec AAAI 2022 repo) rather than imported from
``baselines/ts2vec/``, so this wrapper is self-contained and survives
clean ``setup_baselines.sh`` re-runs.
"""

import os
import numpy as np
from baselines._import_helper import isolated_import
from baselines._epoch_hook import install_epoch_hook

_REPO_DIR = os.path.join(os.path.dirname(__file__), "AutoTCL")
_mod = isolated_import(_REPO_DIR, "AutoTCL_CoST")
AutoTCL = _mod.AutoTCL


class AutoTCLTrainer:

    def __init__(self, input_dims, output_dims=320, hidden_dims=64, depth=10,
                 device='cuda', lr=1e-3, batch_size=16,
                 max_train_length=None, **kwargs):
        # forecast_mode kwarg accepted for backward compatibility with runner
        # call sites, but ignored — single unified config across cls/fcst/ano.
        kwargs.pop("forecast_mode", None)

        self.output_dims = output_dims
        self.max_train_length = max_train_length

        # Unified HP set (AutoTCL_CoST.py class defaults, TS2Vec convention).
        # Paper App D.1: cls "we followed the experiment setting of TS2Vec";
        # paper App D.3.1 Table 7: TS2Vec backbone validated for fcst too.
        # No task-specific branching — same lr / augmask / hard_mask across
        # cls / fcst / ano.
        meta_lr_use = kwargs.pop("meta_lr", 0.01)
        aug_depth = kwargs.pop("aug_depth", 3)
        gumbel_bias = kwargs.pop("gumbel_bias", 0.001)
        gamma_zeta = kwargs.pop("gamma_zeta", 0.05)
        augmask_mode = kwargs.pop("augmask_mode", "binomial")
        hard_mask = kwargs.pop("hard_mask", True)
        # Upstream AutoTCL.fit() defaults (lcoal_weight=0.1, reg_weight=0.001,
        # regular_weight=0.001) — pass-through, no fairness override needed
        # since they apply uniformly to all tasks under unified TS2Vec backbone.
        self._fit_kwargs = {}

        self._model = AutoTCL(
            input_dims=input_dims, output_dims=output_dims,
            hidden_dims=hidden_dims, aug_depth=aug_depth,
            device=device, lr=lr,
            meta_lr=meta_lr_use,
            aug_dim=kwargs.get("aug_dim", 16),
            batch_size=batch_size,
            max_train_length=max_train_length,
            kernels=kwargs.get("kernels", [1, 2, 4, 8, 16, 32, 64, 128]),
            gumbel_bias=gumbel_bias,
            gamma_zeta=gamma_zeta,
            augmask_mode=augmask_mode,
            hard_mask=hard_mask,
        )

        # Swap BOTH _net and augsharenet to TS2Vec backbone (all tasks).
        # Paper §4.2 "We use TS2Vec network architecture" + §4.1 "Augmentation
        # network has the same feature extract architecture". The small
        # projectors (factor_augnet, augmentation_projector) are Linear+Sigmoid
        # downstream of augsharenet output dim, untouched by the swap.
        import torch
        from .ts2vec_backbone import TSEncoder as _TS2VecEncoder
        hidden_for_ts2vec = hidden_dims if isinstance(hidden_dims, int) else 64
        new_net = _TS2VecEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_for_ts2vec,
            depth=depth,
        ).to(device)
        self._model._net = new_net
        self._model.net = torch.optim.swa_utils.AveragedModel(new_net)
        self._model.net.update_parameters(new_net)
        new_aug = _TS2VecEncoder(
            input_dims=input_dims,
            output_dims=kwargs.get("aug_dim", 16),
            hidden_dims=hidden_for_ts2vec,
            depth=aug_depth,
            mask_mode=augmask_mode,
        ).to(device)
        self._model.augsharenet = new_aug
        # Invalidate cached optimizers so AutoTCL.fit() rebuilds both against
        # the new _net AND augsharenet parameters on next call.
        self._model._optimizer = None
        self._model._meta_optimizer = None

        install_epoch_hook(self._model)
        self._callback = None

        def _hook(model, ep, loss):
            if self._callback is not None:
                self._callback(self, ep, loss)
        self._model._after_epoch_callback = _hook

    def fit(self, train_data, epochs, verbose=False, callback=None):
        # Single continuous fit() call; per-epoch callback fires via the
        # n_epochs __setattr__ hook installed in __init__. This avoids
        # upstream's _meta_optimizer reset on each fit() (AutoTCL_CoST.py:253
        # reassigns without hasattr check) that occurred under the previous
        # incremental fit(1)×N pattern.
        self._callback = callback
        target = self._model.n_epochs + epochs
        self._model.fit(train_data, n_epochs=target, valid_dataset=None,
                        **self._fit_kwargs)
        self._callback = None
        return self

    def encode(self, data, raw=False):
        if raw:
            return self._encode_raw(np.nan_to_num(data).astype(np.float32))
        return self._model.encode(np.nan_to_num(data))

    def _encode_raw(self, data):
        """Return (N, T, D) pre-pooling representations using SWA network."""
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(torch.from_numpy(data).float())
        loader = DataLoader(dataset, batch_size=self._model.batch_size)
        org_training = self._model.net.training
        self._model.net.eval()
        outputs = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(self._model.device)
                out = self._model.net(x, None)  # (B, T, D) — SWA network
                outputs.append(out.cpu().numpy())
        self._model.net.train(org_training)
        return np.concatenate(outputs, axis=0)

    @property
    def device(self):
        return self._model.device

    @property
    def model(self):
        return self._model
