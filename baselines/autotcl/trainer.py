"""AutoTCL trainer wrapper.

Uses the TS2Vec encoder (ts2vec_backbone.py) for all three tasks, as in the
paper's classification setting (Sec 4.2, App D.1; App D.3.1 Table 7 reports it
for forecasting). Hyperparameters are the AutoTCL_CoST.py class defaults
(lr=1e-3, augmask='binomial', hard_mask=True, aug_depth=3, gumbel_bias=0.001,
gamma_zeta=0.05), the same for every task.
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

        meta_lr_use = kwargs.pop("meta_lr", 0.01)
        aug_depth = kwargs.pop("aug_depth", 3)
        gumbel_bias = kwargs.pop("gumbel_bias", 0.001)
        gamma_zeta = kwargs.pop("gamma_zeta", 0.05)
        augmask_mode = kwargs.pop("augmask_mode", "binomial")
        hard_mask = kwargs.pop("hard_mask", True)
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

        # Replace both _net and augsharenet with the TS2Vec encoder
        # (paper Sec 4.1/4.2); the small projectors after them are unchanged.
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
        # One continuous fit(); the per-epoch callback fires through the
        # n_epochs hook installed in __init__.
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
