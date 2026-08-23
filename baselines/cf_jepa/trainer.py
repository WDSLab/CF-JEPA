"""
CF-JEPA trainer wrapper for cf_jepa.

Model code: encoder.py, losses.py, cf_jepa.py.
This file provides the cf_jepa-compatible interface:
  - fit(train_data, epochs, verbose, callback)
  - encode(data, raw=False, use_target=False)

CF-JEPA samples crops, splits the future half into three zones, and trains one
zone-conditional linear predictor per zone against the EMA target encoder,
with a horizon-annealed weight of 1 - epoch / total_epochs.
"""

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .losses import variance_reg, covariance_reg
from .cf_jepa import PARAMS_128, PARAMS_320, LinearPredictor, sample_crop




def _get_default_params(output_dims=320):
    """Return Optuna-tuned params for given output dimension."""
    if output_dims == 128:
        return dict(PARAMS_128)
    return dict(PARAMS_320)


class CFJEPATrainer:
    """CF-JEPA trainer with cf_jepa-compatible interface."""

    def __init__(self, input_dims, output_dims=320, hidden_dims=256, depth=5,
                 device='cuda', lr=None, batch_size=16,
                 max_train_length=None, **kwargs):
        self.input_dims = input_dims
        self.device = device
        self.output_dims = output_dims
        self.batch_size = batch_size

        self.params = _get_default_params(output_dims)
        self.params.update(
            enc_hidden=hidden_dims,
            enc_output=output_dims,
            enc_depth=depth,
        )
        if lr is not None:
            self.params['lr'] = lr
        for k, v in kwargs.items():
            if k in self.params:
                self.params[k] = v

        self.max_train_length = max_train_length
        self.encoder = None
        self.target_encoder = None
        self._predictor = None  # Single nn.Module (or None for 'none')
        self._train_mean = None
        self._train_std = None
        self._epoch_counter = 0

    def _build_predictor(self, enc_output):
        """One zone-conditional linear predictor per future zone."""
        return nn.ModuleList([LinearPredictor(dim=enc_output) for _ in range(3)])

    def fit(self, train_data, epochs, verbose=False, callback=None):
        if train_data.ndim == 2:
            train_data = train_data[:, :, np.newaxis]

        N, T, C = train_data.shape
        batch_size = min(self.batch_size, N)
        device = self.device

        train_data = np.nan_to_num(train_data, nan=0.0)
        self._train_mean = train_data.mean(axis=(0, 1), keepdims=True)
        self._train_std = train_data.std(axis=(0, 1), keepdims=True) + 1e-8

        if self.max_train_length is not None and T > self.max_train_length:
            sections = T // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(
                    np.array_split(train_data[:, :sections * self.max_train_length, :],
                                   sections, axis=1),
                    axis=0
                )
                N, T, C = train_data.shape

        train_normed = (train_data - self._train_mean) / self._train_std

        p = self.params
        crop_min, crop_max = p['crop_min'], p['crop_max']
        if T < 50:
            crop_min = max(crop_min, 0.6)
            crop_max = max(crop_max, 0.8)

        n_crops = p['n_crops']
        n_bins = p['n_bins']
        enc_output = p['enc_output']
        lambda_var = p['lambda_var']
        lambda_cov = p['lambda_cov']
        lambda_inv = p['lambda_inv']
        ema_base = p['ema_decay']

        if self.encoder is None:
            self.encoder = Encoder(
                input_dims=C, hidden=p['enc_hidden'],
                output=enc_output, depth=p['enc_depth'],
                n_bins=n_bins,
            ).to(device)
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_encoder.eval()
            for param in self.target_encoder.parameters():
                param.requires_grad_(False)

            self._predictor = self._build_predictor(enc_output).to(device)
            self._all_params = (list(self.encoder.parameters()) +
                                list(self._predictor.parameters()))

            self._optimizer = torch.optim.AdamW(
                self._all_params, lr=p['lr'], weight_decay=0.05)
            self._total_epochs = epochs
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=epochs, eta_min=1e-5)
        else:
            new_total = self._epoch_counter + epochs
            if new_total > self._total_epochs:
                self._total_epochs = new_total
                self._scheduler.T_max = new_total

        x_all = torch.from_numpy(train_normed).float().to(device)

        n_real_batches = max((N - batch_size + 1 + batch_size - 1) // batch_size, 1)
        accum_steps = max(64 // (n_real_batches * batch_size * n_crops), 1)

        for ep in range(1, epochs + 1):
            self._epoch_counter += 1
            ema_cur = 1 - (1 - ema_base) * (np.cos(np.pi * ep / epochs) + 1) / 2

            self.encoder.train()
            self._predictor.train()

            perm = torch.randperm(N)
            epoch_loss = 0
            n_batches = 0

            for i in range(0, N - batch_size + 1, batch_size):
                idx = perm[i:i + batch_size]
                x = x_all[idx]

                with torch.no_grad():
                    z_full_target = self.target_encoder(x)

                loss_pred_total = 0.0
                n_pred_terms = 0
                z_crops_for_vicreg = []
                z_crops_raw = []

                total_crops = n_crops * accum_steps
                for _ in range(total_crops):
                    crop_start, crop_end = sample_crop(T, crop_min, crop_max)
                    crop_len = crop_end - crop_start

                    x_crop = x[:, crop_start:crop_end, :]
                    z_crop = self.encoder(x_crop)
                    z_crop_pooled = F.adaptive_avg_pool1d(
                        z_crop.transpose(1, 2), n_bins
                    ).transpose(1, 2)

                    # CF-JEPA: 3 zone-conditional predictors; the future half
                    # is split into 3 zones and each predictor targets its zone.
                    remaining = T - crop_end
                    zone_size = remaining // 3
                    for zi in range(3):
                        pred_zi = self._predictor[zi](z_crop)
                        zone_start = crop_end + zi * zone_size
                        zone_end = (crop_end + (zi + 1) * zone_size
                                    if zi < 2 else T)
                        avail = zone_end - zone_start
                        if avail < 1:
                            continue
                        if avail >= crop_len:
                            max_s = zone_end - crop_len
                            h_start = np.random.randint(
                                zone_start, max(max_s, zone_start) + 1)
                            h_end = h_start + crop_len
                            target_h = z_full_target[:, h_start:h_end, :]
                            loss_pred_total = loss_pred_total + F.l1_loss(
                                F.normalize(pred_zi, dim=-1),
                                F.normalize(target_h, dim=-1))
                        else:
                            target_h = z_full_target[:, zone_start:zone_end, :]
                            pred_pooled = F.adaptive_avg_pool1d(
                                pred_zi.transpose(1, 2), avail
                            ).transpose(1, 2)
                            loss_pred_total = loss_pred_total + F.l1_loss(
                                F.normalize(pred_pooled, dim=-1),
                                F.normalize(target_h, dim=-1))
                        n_pred_terms += 1

                    z_crops_for_vicreg.append(z_crop_pooled.mean(dim=1))
                    z_crops_raw.append(z_crop)

                if n_pred_terms > 0:
                    loss_pred_total = loss_pred_total / n_pred_terms

                if z_crops_for_vicreg:
                    z_cat = torch.cat(z_crops_for_vicreg, dim=0)
                    var_loss = variance_reg(z_cat)
                    cov_loss = covariance_reg(z_cat)
                else:
                    var_loss = torch.tensor(0.0, device=device)
                    cov_loss = torch.tensor(0.0, device=device)

                # Multi-scale invariance requires multiple crop views from the
                # same sequence, supplied by z_crops_raw.
                #
                # NOTE on the pool_size loop: each pooled crop is collapsed back
                # to (B, D) by the trailing .mean(dim=1) below, so the three pool
                # sizes yield (near-)identical vectors — exactly identical when
                # the crop length is divisible by the pool size. The term is
                # therefore crop-level invariance (agreement between the mean-
                # pooled representations of crops from the same series), not
                # per-bin agreement at three temporal resolutions. Retaining the
                # bins would require aligning bins across crops, which is not
                # well defined here because crops cover different, arbitrarily
                # positioned spans of the series. Kept as-is: all reported
                # numbers were produced with this formulation.
                if z_crops_raw:
                    inv_loss = 0
                    min_crop_len = min(z.shape[1] for z in z_crops_raw)
                    pool_sizes = [ps for ps in [2, 4, 8] if ps <= min_crop_len]
                    if not pool_sizes:
                        pool_sizes = [1]
                    for pool_size in pool_sizes:
                        pools = []
                        for z_raw in z_crops_raw:
                            pp = F.adaptive_avg_pool1d(
                                z_raw.transpose(1, 2), pool_size
                            ).transpose(1, 2).mean(dim=1)
                            pools.append(pp)
                        z_stack = torch.stack(pools, dim=0)
                        z_mean = z_stack.mean(dim=0, keepdim=True)
                        inv_loss = inv_loss + ((z_stack - z_mean) ** 2).mean()
                    inv_loss = inv_loss / len(pool_sizes)
                else:
                    inv_loss = torch.tensor(0.0, device=device)

                # Horizon annealing: the prediction term is down-weighted as
                # training proceeds.
                horizon_weight = 1.0 - (self._epoch_counter / self._total_epochs)
                loss = (lambda_var * var_loss
                        + lambda_cov * cov_loss
                        + lambda_inv * inv_loss
                        + horizon_weight * loss_pred_total)

                # NaN/Inf guard: skip step if loss diverged this batch.
                # Preserves encoder weights from the last good step instead of
                # corrupting them via backward. Standard safeguard for SSL
                # training, where rare numerical instabilities can occur on
                # long sequences (e.g. ETTm at T=2048 chunks).
                if not torch.isfinite(loss):
                    self._nan_skip_count = getattr(self, '_nan_skip_count', 0) + 1
                    if verbose and self._nan_skip_count <= 5:
                        print(f"  [WARN] non-finite loss at epoch "
                              f"{self._epoch_counter}, batch start {i}, "
                              f"skipping step (cumulative skips: "
                              f"{self._nan_skip_count})")
                    continue

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._all_params, 1.0)
                self._optimizer.step()

                with torch.no_grad():
                    # EMA over PARAMETERS only (weight/bias and BatchNorm affine
                    # gamma/beta). This matches the reference momentum-encoder
                    # implementations (MoCo, BYOL), which likewise EMA parameters
                    # and not buffers.
                    #
                    # One consequence is specific to this code and worth stating
                    # plainly. BatchNorm running statistics are buffers, so they
                    # are outside this loop; in the reference implementations the
                    # target still fills them by running its forward pass in train
                    # mode. Here the target is deep-copied before training and kept
                    # in eval() (see __init__ and encode()), so its running stats
                    # stay at their initial values (mean 0, var 1) and the target's
                    # BatchNorm reduces to an identity normalization followed by
                    # the EMA-tracked affine transform.
                    #
                    # All reported numbers were produced with this configuration.
                    # Switching the target to train mode (the reference behavior)
                    # was checked separately and was not adopted: it changes the
                    # target representation and the reported forecasting/anomaly
                    # numbers do not reproduce under it. Classification is
                    # unaffected, as it is evaluated with the online encoder.
                    for tp, pp in zip(self.target_encoder.parameters(),
                                      self.encoder.parameters()):
                        tp.data.mul_(ema_cur).add_(pp.data, alpha=1 - ema_cur)

                epoch_loss += loss.item() if torch.is_tensor(loss) else float(loss)
                n_batches += 1

            self._scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            if callback is not None:
                callback(self, self._epoch_counter, avg_loss)
            if verbose and (ep % 10 == 0 or ep == 1):
                print(f"  Epoch {self._epoch_counter:3d}: loss={avg_loss:.4f}")

        return self

    def encode(self, data, raw=False, use_target=False):
        if self.encoder is None:
            raise RuntimeError("Call fit() first")

        enc = self.target_encoder if use_target else self.encoder
        was_training = enc.training
        enc.eval()
        try:
            if data.ndim == 2:
                data = data[:, :, np.newaxis]

            data = np.nan_to_num(data, nan=0.0)
            data = (data - self._train_mean) / self._train_std

            N = data.shape[0]
            batch_size = min(self.batch_size, N)
            results = []

            with torch.no_grad():
                for i in range(0, N, batch_size):
                    x = torch.from_numpy(data[i:i+batch_size]).float().to(self.device)
                    z = enc(x)

                    if raw:
                        results.append(z.cpu().numpy())
                    else:
                        n_bins = self.params['n_bins']
                        z_pooled = F.adaptive_avg_pool1d(
                            z.transpose(1, 2), n_bins
                        ).transpose(1, 2)
                        results.append(z_pooled.reshape(z.shape[0], -1).cpu().numpy())

            return np.concatenate(results, axis=0)
        finally:
            if was_training:
                enc.train()
