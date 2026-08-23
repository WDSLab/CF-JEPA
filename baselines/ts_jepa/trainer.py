"""
TS-JEPA trainer wrapper.

Wraps the Encoder and Predictor from TS_JEPA to expose the unified
cf_jepa trainer interface.  Pre-trains a JEPA (Joint-Embedding
Predictive Architecture) with patch-level masking and EMA target encoder.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from baselines._import_helper import isolated_import

# ---------------------------------------------------------------------------
# Import original TS-JEPA components
# ---------------------------------------------------------------------------
_REPO_DIR = os.path.join(os.path.dirname(__file__), "TS_JEPA")
_enc_mod = isolated_import(_REPO_DIR, "src.models.encoder")
_pred_mod = isolated_import(_REPO_DIR, "src.models.predictor")
_mask_mod = isolated_import(_REPO_DIR, "src.models.utils.mask_utils")

Encoder = _enc_mod.Encoder
Predictor = _pred_mod.Predictor
apply_mask = _mask_mod.apply_mask


class TSJEPATrainer:
    """Unified trainer wrapper around TS-JEPA."""

    def __init__(
        self,
        input_dims,
        output_dims=384,
        hidden_dims=64,
        depth=10,
        device="cuda",
        lr=6e-5,
        batch_size=16,
        max_train_length=None,
        **kwargs,
    ):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length

        # Architecture hyper-parameters (defaults from original config_pretrain.py)
        self.embed_dim = kwargs.get("embed_dim", 384)
        self.nhead = kwargs.get("nhead", 6)
        # Ensure embed_dim is divisible by nhead and even
        if self.embed_dim % self.nhead != 0:
            self.embed_dim = (self.embed_dim // self.nhead) * self.nhead
        if self.embed_dim < self.nhead:
            self.embed_dim = self.nhead * 2
        if self.embed_dim % 2 != 0:
            self.embed_dim += 1
        self.num_layers = kwargs.get("num_layers", 3)
        self.kernel_size = kwargs.get("kernel_size", 3)
        self.predictor_embed = kwargs.get("predictor_embed", 128)
        self.predictor_nhead = kwargs.get("predictor_nhead", 2)
        # Ensure predictor_embed is divisible by predictor_nhead and even
        if self.predictor_embed % self.predictor_nhead != 0:
            self.predictor_embed = (
                (self.predictor_embed // self.predictor_nhead) * self.predictor_nhead
            )
        if self.predictor_embed < self.predictor_nhead:
            self.predictor_embed = self.predictor_nhead * 2
        if self.predictor_embed % 2 != 0:
            self.predictor_embed += self.predictor_nhead
        self.predictor_layers = kwargs.get("predictor_layers", 1)
        self.patch_size = kwargs.get("patch_size", 32)
        self.mask_ratio = kwargs.get("mask_ratio", 0.7)
        self.ema_momentum = kwargs.get("ema_momentum", 0.998)
        self.clip_grad = kwargs.get("clip_grad", 10.0)

        # Models are built lazily in fit() when we know T
        self._net = None
        self._predictor = None
        self._net_ema = None
        self.actual_patch_size = None
        self.num_patches = None

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(model):
        """Initialize weights following the original TS-JEPA (trunc_normal for Linear, constant for LayerNorm)."""
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------
    def _build_models(self, T):
        """Construct encoder, predictor, and EMA encoder."""
        D = self.input_dims
        ps = min(self.patch_size, max(T // 4, 1))
        self.actual_patch_size = ps
        self.num_patches = (T + ps - 1) // ps  # ceil division
        dim_in = ps * D

        self._net = Encoder(
            num_patches=self.num_patches,
            dim_in=dim_in,
            kernel_size=self.kernel_size,
            embed_dim=self.embed_dim,
            embed_bias=True,
            nhead=self.nhead,
            num_layers=self.num_layers,
            jepa=True,
        ).to(self.device)

        self._predictor = Predictor(
            num_patches=self.num_patches,
            encoder_embed_dim=self.embed_dim,
            predictor_embed_dim=self.predictor_embed,
            nhead=self.predictor_nhead,
            num_layers=self.predictor_layers,
        ).to(self.device)

        # Weight initialization (matches original: trunc_normal for Linear, constant for LayerNorm)
        self._init_weights(self._net)
        self._init_weights(self._predictor)

        self._net_ema = copy.deepcopy(self._net)
        for p in self._net_ema.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    def _patchify(self, data):
        """Convert (N, T, D) numpy to (N, num_patches, patch_size*D) tensor."""
        N, T, D = data.shape
        ps = self.actual_patch_size
        # Pad T to be a multiple of ps
        pad_len = self.num_patches * ps - T
        if pad_len > 0:
            data = np.pad(data, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
        # Reshape: (N, num_patches, ps, D) -> (N, num_patches, ps*D)
        data = data.reshape(N, self.num_patches, ps, D)
        data = data.reshape(N, self.num_patches, ps * D)
        return torch.from_numpy(data).float()

    # ------------------------------------------------------------------
    def _generate_masks(self, batch_size):
        """Generate random mask / non-mask index tensors.

        Returns
        -------
        masks : list of 1 tensor, shape (B, num_masked)
        non_masks : list of 1 tensor, shape (B, num_unmasked)

        The lists wrap single tensors because ``apply_mask`` expects a list.
        """
        num_masked = max(1, int(self.num_patches * self.mask_ratio))
        num_unmasked = self.num_patches - num_masked

        masks_list = []
        non_masks_list = []
        for _ in range(batch_size):
            perm = torch.randperm(self.num_patches)
            non_masks_list.append(perm[:num_unmasked].sort().values)
            masks_list.append(perm[num_unmasked:].sort().values)

        # apply_mask expects a *list* of tensors with shape (B, K)
        masks = [torch.stack(masks_list)]           # [(B, num_masked)]
        non_masks = [torch.stack(non_masks_list)]   # [(B, num_unmasked)]
        return masks, non_masks

    # ------------------------------------------------------------------
    def fit(self, train_data, epochs, verbose=False, callback=None):
        """Pre-train the TS-JEPA encoder.

        Parameters
        ----------
        train_data : np.ndarray, shape (N, T, D)
        epochs : int
        verbose : bool
        callback : callable(trainer, epoch, loss) or None

        Returns
        -------
        self
        """
        N, T, D = train_data.shape
        data = np.nan_to_num(train_data, nan=0.0).astype(np.float32)

        # Chunk N=1 long fcst series into exact max_train_length pieces:
        # bounds num_patches (else Traffic/weather OOM the host) and gives N>1.
        if self.max_train_length is not None and T > self.max_train_length:
            n_full = T // self.max_train_length
            chunks = [
                data[:, i * self.max_train_length:(i + 1) * self.max_train_length]
                for i in range(n_full)
            ]
            data = np.concatenate(chunks, axis=0)
            T = self.max_train_length
            N = data.shape[0]

        if self._net is None:
            self._build_models(T)
            self._optimizer = None
            self._epoch_counter = 0

        patches = self._patchify(data)
        dataset = TensorDataset(patches)
        loader = DataLoader(
            dataset,
            batch_size=min(self.batch_size, N),
            shuffle=True,
            drop_last=True,
        )

        # Create optimizer and scheduler only once
        if self._optimizer is None:
            param_groups = [
                {"params": self._net.parameters()},
                {"params": self._predictor.parameters()},
            ]
            self._optimizer = torch.optim.AdamW(param_groups, lr=self.lr)
            self._scheduler = torch.optim.lr_scheduler.LinearLR(
                self._optimizer, start_factor=1.0, end_factor=0.5, total_iters=epochs,
            )
            self._total_epochs = epochs

        # ipe_scale follows upstream pretrain.py:128-134. The factor keeps the
        # target encoder's EMA momentum below 1.0 throughout training so it
        # always tracks the online encoder; without it m hits 1.0 and the
        # target freezes mid-training, breaking the JEPA prediction signal.
        ipe_scale = 1.25

        for epoch in range(1, epochs + 1):
            self._epoch_counter += 1
            # Per-epoch EMA momentum, sampled once and reused across all batches
            # in this epoch (matches upstream pretrain.py:146 `m = next(ema_scheduler)`).
            m = self.ema_momentum + (self._epoch_counter - 1) * (
                1.0 - self.ema_momentum
            ) / (self._total_epochs * ipe_scale)
            self._net.train()
            self._predictor.train()
            cum_loss = 0.0
            n_iters = 0

            for (batch_data,) in loader:
                x = batch_data.to(self.device)  # (B, num_patches, ps*D)
                B = x.size(0)

                masks, non_masks = self._generate_masks(B)
                masks = [m.to(self.device) for m in masks]
                non_masks = [m.to(self.device) for m in non_masks]

                self._optimizer.zero_grad()

                # Target from EMA encoder (full sequence, no masking)
                with torch.no_grad():
                    target = self._net_ema(x)  # (B, num_patches, embed_dim)
                    target = F.layer_norm(target, (target.size(-1),))
                    target_masked = apply_mask(target, masks)  # (B, num_masked, embed_dim)

                # Encode only unmasked patches
                tokens = self._net(x, mask=non_masks)  # (B, num_unmasked, embed_dim)

                # Predict masked from unmasked
                pred = self._predictor(
                    tokens,
                    mask=masks[0],
                    non_masks=non_masks[0],
                )  # (B, num_masked, embed_dim)

                loss = torch.mean(torch.abs(pred - target_masked))
                loss.backward()
                if self.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(self._net.parameters(), self.clip_grad)
                    torch.nn.utils.clip_grad_norm_(self._predictor.parameters(), self.clip_grad)
                self._optimizer.step()

                # EMA update — same m for all batches in this epoch (set above).
                with torch.no_grad():
                    for p_q, p_k in zip(
                        self._net.parameters(), self._net_ema.parameters()
                    ):
                        p_k.data.mul_(m).add_((1.0 - m) * p_q.detach().data)

                cum_loss += loss.item()
                n_iters += 1

            self._scheduler.step()
            avg_loss = cum_loss / max(n_iters, 1)
            if verbose:
                print(f"Epoch {self._epoch_counter}: loss={avg_loss:.4f}")
            if callback is not None:
                callback(self, self._epoch_counter, avg_loss)

        return self

    # ------------------------------------------------------------------
    def encode(self, data, raw=False):
        """Produce representations.

        Parameters
        ----------
        data : np.ndarray, shape (N, T, D)
        raw : bool, if True return (N, num_patches, embed_dim) without pooling

        Returns
        -------
        np.ndarray, shape (N, embed_dim) or (N, num_patches, embed_dim) if raw
        """
        if self._net is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        data = np.nan_to_num(data, nan=0.0).astype(np.float32)
        N, T, D = data.shape
        max_T = self.num_patches * self.actual_patch_size
        if T > max_T:
            # Chunk along time, encode each chunk, concat along patch dim
            n_chunks = (T + max_T - 1) // max_T
            pad_len = n_chunks * max_T - T
            if pad_len > 0:
                data = np.pad(data, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
            data = data.reshape(N, n_chunks, max_T, D).transpose(1, 0, 2, 3).reshape(n_chunks * N, max_T, D)
            patches = self._patchify(data)
        else:
            n_chunks = 1
            patches = self._patchify(data)
        dataset = TensorDataset(patches)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        was_training = self._net.training
        self._net.eval()
        outputs = []
        try:
            with torch.no_grad():
                for (batch_data,) in loader:
                    x = batch_data.to(self.device)
                    out = self._net(x)  # (B, num_patches, embed_dim)
                    if not raw:
                        out = out.mean(dim=1)  # (B, embed_dim)
                    outputs.append(out.cpu().numpy())
        finally:
            if was_training:
                self._net.train()

        result = np.concatenate(outputs, axis=0)
        if n_chunks > 1:
            # (n_chunks*N, num_patches, D_emb) -> (N, n_chunks*num_patches, D_emb) for raw,
            # or (n_chunks*N, D_emb) -> (N, D_emb) mean across chunks for pooled
            if raw:
                result = result.reshape(n_chunks, N, self.num_patches, -1).transpose(1, 0, 2, 3).reshape(N, n_chunks * self.num_patches, -1)
            else:
                result = result.reshape(n_chunks, N, -1).mean(axis=0)
        return result

    # ------------------------------------------------------------------
    @property
    def model(self):
        return self._net
