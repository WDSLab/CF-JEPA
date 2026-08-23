"""InfoTS trainer wrapper.

Includes monkey-patch fix for unsup_pred batch size mismatch on small datasets
(e.g., AtrialFibrillation N=15 < batch_size=16).
"""

import os
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from baselines._import_helper import isolated_import
from baselines._epoch_hook import install_epoch_hook

_REPO_DIR = os.path.join(os.path.dirname(__file__), "InfoTS")
_mod = isolated_import(_REPO_DIR, "infots")
InfoTS = _mod.InfoTS


@contextlib.contextmanager
def _redirect_cuda(target_device):
    """Temporarily redirect Module.cuda() to .to(target_device).

    Upstream `infots.py` hardcodes `.cuda()` for `pred`, `unsup_pred`, and
    `aug` during InfoTS.__init__, ignoring the `device` argument. This
    breaks CPU init (and would break any non-default-cuda device). We
    redirect `.cuda()` to `.to(device)` for the duration of construction
    only; the supervised-path `.cuda()` calls in `fit()` are not relevant
    because we always call `task_type='pretraining', supervised_meta=False`.
    """
    _orig = torch.nn.Module.cuda
    torch.nn.Module.cuda = lambda self, dev=None: self.to(target_device)
    try:
        yield
    finally:
        torch.nn.Module.cuda = _orig


def _patch_infots_small_batch(model):
    """Monkey-patch InfoTS to handle batch_size > N_train.

    Original code creates unsup_pred = Linear(output_dims, batch_size),
    causing mismatch when actual batch < batch_size. This patch dynamically
    resizes unsup_pred to match actual batch size.
    """
    model._unsup_pred_dim = model.batch_size
    model._output_dims = model.unsup_pred.in_features

    original_meta_fit = model.meta_fit
    original_super_fit = model.super_fit

    def _get_unsup_pred(bs):
        if model._unsup_pred_dim != bs:
            model.unsup_pred = nn.Linear(model._output_dims, bs).to(model.device)
            model._unsup_pred_dim = bs
        return model.unsup_pred

    def patched_meta_fit(train_loader, meta_optimizer, meta_beta, supervised_meta, cls_optimizer):
        pre_flag = model._net.training
        model._net.eval()
        for batch in train_loader:
            x = batch[0][0]
            y = batch[1][0]

            if model.max_train_length is not None and x.size(1) > model.max_train_length:
                window_offset = np.random.randint(x.size(1) - model.max_train_length + 1)
                x = x[:, window_offset: window_offset + model.max_train_length]
            x = x.to(model.device)
            if supervised_meta:
                y = y.to(model.device)
            else:
                y = torch.arange(x.size(0), dtype=torch.int64).to(model.device)

            meta_optimizer.zero_grad()
            outv, outx = model.get_features(x)

            from infots import global_infoNCE
            MI_vx_loss = global_infoNCE(outv, outx)

            zv = F.max_pool1d(outv.transpose(1, 2).contiguous(), kernel_size=outv.size(1)).transpose(1, 2)
            zx = F.max_pool1d(outx.transpose(1, 2).contiguous(), kernel_size=outx.size(1)).transpose(1, 2)

            if supervised_meta:
                pred_yv = torch.squeeze(model.pred(zv), 1)
                pred_yx = torch.squeeze(model.pred(zx), 1)
            else:
                unsup = _get_unsup_pred(x.size(0))
                pred_yv = torch.squeeze(unsup(zv), 1)
                pred_yx = torch.squeeze(unsup(zx), 1)

            MI_vy_loss = model.CE(pred_yv, y)
            MI_xy_loss = model.CE(pred_yx, y)

            vx_vy_loss = meta_beta * (MI_vy_loss + MI_xy_loss)
            vx_vy_loss.backward()
            meta_optimizer.step()
            cls_optimizer.step()

        if pre_flag:
            model._net.train()

    def patched_super_fit(train_loader, supervised_meta, cls_optimizer):
        for batch in train_loader:
            x = batch[0][0]
            y = batch[1][0]

            if model.max_train_length is not None and x.size(1) > model.max_train_length:
                window_offset = np.random.randint(x.size(1) - model.max_train_length + 1)
                x = x[:, window_offset: window_offset + model.max_train_length]
            x = x.to(model.device)
            if supervised_meta:
                y = y.to(model.device)
            else:
                y = torch.arange(x.size(0), dtype=torch.int64).to(model.device)

            cls_optimizer.zero_grad()
            outx = model._net(x)
            zx = F.max_pool1d(outx.transpose(1, 2).contiguous(), kernel_size=outx.size(1)).transpose(1, 2)

            if supervised_meta:
                pred_yx = torch.squeeze(model.pred(zx), 1)
            else:
                unsup = _get_unsup_pred(x.size(0))
                pred_yx = torch.squeeze(unsup(zx), 1)

            celoss = model.CE(pred_yx, y)
            celoss.backward()
            cls_optimizer.step()

    model.meta_fit = patched_meta_fit
    model.super_fit = patched_super_fit


class InfoTSTrainer:

    def __init__(self, input_dims, output_dims=320, hidden_dims=60, depth=10,
                 device='cuda', lr=0.001, batch_size=16,
                 max_train_length=None, **kwargs):
        self.output_dims = output_dims
        self.max_train_length = max_train_length

        with _redirect_cuda(device):
            self._model = InfoTS(
                input_dims=input_dims, output_dims=output_dims,
                hidden_dims=hidden_dims, num_cls=kwargs.get("num_cls", 2),
                depth=depth, device=device, lr=lr,
                meta_lr=kwargs.get("meta_lr", 0.01),
                batch_size=batch_size,
                max_train_length=max_train_length,
                mask_mode=kwargs.get("mask_mode", "binomial"),
                dropout=kwargs.get("dropout", 0.1),
                aug_p1=kwargs.get("aug_p1", 0.2),
                aug_p2=kwargs.get("aug_p2", 0.0),
                eval_every_epoch=kwargs.get("eval_every_epoch", 999999),
                used_augs=kwargs.get("used_augs"),
            )
        _patch_infots_small_batch(self._model)
        install_epoch_hook(self._model)
        self._callback = None

        def _hook(model, ep, loss):
            if self._callback is not None:
                self._callback(self, ep, loss)
        self._model._after_epoch_callback = _hook

    def fit(self, train_data, epochs, verbose=False, callback=None):
        self._callback = callback
        target = self._model.n_epochs + epochs
        self._model.fit(train_data, n_epochs=target, verbose=False,
                        supervised_meta=False, train_labels=None,
                        valid_dataset=None, task_type='pretraining')
        self._callback = None
        return self

    def encode(self, data, raw=False):
        if raw:
            return self._encode_raw(np.nan_to_num(data, nan=0.0).astype(np.float32))
        return self._model.encode(np.nan_to_num(data, nan=0.0).astype(np.float32))

    def _encode_raw(self, data):
        """Return (N, T, D) pre-pooling representations."""
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(torch.from_numpy(data).float())
        loader = DataLoader(dataset, batch_size=self._model.batch_size)
        was_training = self._model.net.training
        self._model.net.eval()
        outputs = []
        try:
            with torch.no_grad():
                for batch in loader:
                    x = batch[0].to(self._model.device)
                    out = self._model.net(x, None)
                    outputs.append(out.cpu().numpy())
        finally:
            if was_training:
                self._model.net.train()
        return np.concatenate(outputs, axis=0)

    @property
    def _net(self):
        return self._model._net

    @property
    def device(self):
        return self._model.device

    @property
    def model(self):
        return self._model
