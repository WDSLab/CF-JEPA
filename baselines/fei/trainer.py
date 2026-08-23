"""
FEI (Frequency-masked Embedding Inference) trainer wrapper.

Implements the FEI pre-training loop using the ResNet encoder and
frequency-domain masking strategy from the original codebase, with inline
model definitions to avoid the TrainableModule dependency chain.
"""

import os
import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from baselines._import_helper import isolated_import

# ---------------------------------------------------------------------------
# Import mask-generation and mask-application utilities from the original repo
# ---------------------------------------------------------------------------
_REPO_DIR = os.path.join(
    os.path.dirname(__file__), "Frequency-masked-Embedding-Inference",
)
_util_mod = isolated_import(_REPO_DIR, "util.utils")

get_batch_continuous_freq_mask = _util_mod.get_batch_continuous_freq_mask
get_batch_discrete_freq_mask = _util_mod.get_batch_discrete_freq_mask
apply_freq_reduce_mask = _util_mod.apply_freq_reduce_mask


# ===================================================================
# Inline model components (avoiding TrainableModule import chain)
# ===================================================================

class _Transpose(nn.Module):
    """Swap the last two dimensions."""
    def forward(self, x):
        return x.transpose(-1, -2)


class _ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1,
                 out_act=True):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.GELU(),
            nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size,
                      stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channel),
            )
        self.out_act = nn.GELU() if out_act else None

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        if self.out_act is not None:
            out = self.out_act(out)
        return out


class _ResNet(nn.Module):
    """1-D ResNet matching FEI's 'big' size (paper default).

    Channels: 16 -> 128 -> 256 -> 512 -> 1024. The original FEIConfig sets
    encoder_size='big' which selects this 1024-dim final layer; an earlier
    version of this wrapper used 'tiny' (128-dim) for representation-size
    alignment with SimMTM, but that downsized FEI's capacity ~50x relative
    to its paper configuration.
    """

    def __init__(self, stride=2):
        super().__init__()
        self.in_channel = 16
        self.layer1 = self._make_layer(128, 2, stride=stride)
        self.layer2 = self._make_layer(256, 2, stride=stride)
        self.layer3 = self._make_layer(512, 2, stride=stride)
        self.layer4 = self._make_layer(1024, 2, stride=stride, out_act=True)

    def _make_layer(self, channels, num_blocks, stride, out_act=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_ResBlock(self.in_channel, channels, stride=s,
                                    out_act=out_act))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, L, 16) -> (B, L', 1024)"""
        x = x.transpose(-1, -2)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x.transpose(-1, -2)


# ===================================================================
# Trainer
# ===================================================================

class FEITrainer:
    """Unified trainer wrapper around FEI (Frequency-masked Embedding
    Inference).

    The encoder pipeline is:
        input_embedding (Conv1d-based) -> ResNet -> mean-pool -> hidden_dim

    Pre-training uses frequency-domain masking with an EMA momentum
    encoder and embedding / mask prediction losses.
    """

    # FEI's 'big' ResNet (paper default) has a fixed 1024-channel final
    # layer, so output_dims is not user-configurable.
    _FIXED_OUTPUT_DIMS = 1024

    def __init__(
        self,
        input_dims,
        output_dims=_FIXED_OUTPUT_DIMS,
        hidden_dims=64,
        depth=10,
        device="cuda",
        lr=0.0002,
        batch_size=16,
        max_train_length=None,
        **kwargs,
    ):
        if output_dims != self._FIXED_OUTPUT_DIMS:
            warnings.warn(
                f"FEI architecture produces {self._FIXED_OUTPUT_DIMS}-dim "
                f"representations (tiny ResNet final layer); requested "
                f"output_dims={output_dims} is ignored.",
                stacklevel=2,
            )
        self.input_dims = input_dims
        self.hidden_dim = self._FIXED_OUTPUT_DIMS
        self.output_dims = self.hidden_dim
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length

        # FEI-specific hyper-parameters
        self.ema_rate = kwargs.get("ema_rate", 0.995)
        self.mask_ratio = kwargs.get("mask_ratio", (0.0, 0.7))
        self.target_num = kwargs.get("target_num", 1)
        self.mask_type = kwargs.get("mask_type", "discrete")
        self.mask_pool_size = kwargs.get("mask_pool_size", 1000)
        self.resnet_stride = kwargs.get("resnet_stride", 2)

        # Models built lazily in fit() when T is known
        self._input_embedding = None
        self._encoder = None
        self._net = None  # alias used by @model property
        self._T = None

    # ------------------------------------------------------------------
    def _build_model(self, T):
        """Construct all FEI sub-modules."""
        D = self.input_dims

        # --- Input embedding: (B, T, D) -> transpose -> Conv1d -> pool -> transpose
        self._input_embedding = nn.Sequential(
            _Transpose(),
            nn.Conv1d(D, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.AvgPool1d(2, 2),
            _Transpose(),
        ).to(self.device)

        # --- ResNet encoder
        self._encoder = _ResNet(stride=self.resnet_stride).to(self.device)

        # --- Feature fusion: mean over temporal dim
        self._feature_fusion = lambda x: x.mean(dim=-2)

        # --- Momentum copies
        self._mom_input_embedding = copy.deepcopy(self._input_embedding)
        self._mom_input_embedding.requires_grad_(False)
        self._mom_encoder = copy.deepcopy(self._encoder)
        self._mom_encoder.requires_grad_(False)

        # --- Subspace mapper
        shrink_dim = self.hidden_dim // 2
        predictor_dim = shrink_dim // 2

        self._subspace_mapper = nn.Linear(self.hidden_dim, shrink_dim).to(
            self.device
        )
        self._mom_subspace_mapper = copy.deepcopy(self._subspace_mapper)
        self._mom_subspace_mapper.requires_grad_(False)

        # --- Embedding predictor
        self._embedding_predictor = nn.Sequential(
            nn.Linear(shrink_dim, predictor_dim),
            nn.GELU(),
            nn.Linear(predictor_dim, shrink_dim),
        ).to(self.device)

        # --- Mask predictor
        self._mask_predictor = nn.Sequential(
            nn.Linear(shrink_dim, predictor_dim),
            nn.GELU(),
            nn.Linear(predictor_dim, shrink_dim),
        ).to(self.device)

        # --- Mask encoder (learnable parameter)
        freq_len = T // 2 + 1
        self._mask_encoder = nn.Parameter(
            torch.randn(freq_len, shrink_dim, device=self.device)
        )

        # Convenience: collect all trainable parameters into a ModuleList
        # so we can pass them to the optimizer.
        self._net = nn.ModuleDict({
            "input_embedding": self._input_embedding,
            "encoder": self._encoder,
            "subspace_mapper": nn.ModuleList([self._subspace_mapper]),
            "embedding_predictor": self._embedding_predictor,
            "mask_predictor": self._mask_predictor,
        }).to(self.device)
        # _mask_encoder is a raw Parameter, handled separately

        self._T = T

    # ------------------------------------------------------------------
    def _mask_encode(self, masks):
        """Encode frequency masks into embeddings.

        Parameters
        ----------
        masks : Tensor, shape (B, target_num, freq_len)

        Returns
        -------
        Tensor, shape (B, target_num, shrink_dim)
        """
        inv_masks = 1.0 - masks  # 1 where masked, 0 where kept
        # Weighted sum of mask_encoder rows
        mask_embed = torch.matmul(inv_masks, self._mask_encoder) / (
            torch.sum(inv_masks, dim=-1, keepdim=True).sqrt() + 1e-6
        )
        return mask_embed

    # ------------------------------------------------------------------
    def _generate_mask_pool(self, pool_size, T):
        """Pre-generate a pool of frequency masks."""
        if self.mask_type == "discrete":
            return get_batch_discrete_freq_mask(
                pool_size, T, list(self.mask_ratio), self.target_num,
            )
        else:
            return get_batch_continuous_freq_mask(
                pool_size, T, list(self.mask_ratio), self.target_num,
            )

    # ------------------------------------------------------------------
    def _select_masks(self, batch_size, mask_pool):
        indices = torch.randperm(mask_pool.size(0))[:batch_size]
        return mask_pool[indices]

    # ------------------------------------------------------------------
    def fit(self, train_data, epochs, verbose=False, callback=None):
        """Pre-train the FEI encoder.

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
        # bounds mask_encoder (Parameter[T//2+1, 512]) and mask_pool size,
        # and gives N>1 real training batches.
        if self.max_train_length is not None and T > self.max_train_length:
            n_full = T // self.max_train_length
            chunks = [
                data[:, i * self.max_train_length:(i + 1) * self.max_train_length]
                for i in range(n_full)
            ]
            data = np.concatenate(chunks, axis=0)
            T = self.max_train_length
            N = data.shape[0]

        if self._net is None or self._T != T:
            self._build_model(T)
            self._optimizer = None
            self._scheduler = None
            self._epoch_counter = 0
            self._momentum_iter = 0

        # Pre-generate mask pool
        mask_pool = self._generate_mask_pool(
            self.batch_size * self.mask_pool_size, T,
        )

        data_t = torch.from_numpy(data)
        dataset = TensorDataset(data_t)
        # drop_last=False: when post-chunking N < batch_size (e.g. ETTh1 fcst
        # T=8640 / mtl=2048 → N_chunks=4), drop_last=True would yield 0 batches
        # and silently skip all training — encoder stays at random init with no
        # error. Same hazard for UEA cls datasets with N_train < 16.
        loader = DataLoader(
            dataset, batch_size=min(self.batch_size, N),
            shuffle=True, drop_last=False,
        )

        # Create optimizer/scheduler only once
        if self._optimizer is None:
            trainable_params = (
                list(self._input_embedding.parameters())
                + list(self._encoder.parameters())
                + list(self._subspace_mapper.parameters())
                + list(self._embedding_predictor.parameters())
                + list(self._mask_predictor.parameters())
                + [self._mask_encoder]
            )
            self._optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer, gamma=0.90,
            )
            self._total_iters = epochs * len(loader)

        criterion = nn.MSELoss()

        for epoch in range(1, epochs + 1):
            self._epoch_counter += 1
            self._net.train()
            cum_loss = 0.0
            n_iters = 0

            for (batch_data,) in loader:
                x = batch_data.to(self.device)  # (B, T, D)
                B = x.size(0)

                masks = self._select_masks(B, mask_pool).to(self.device)
                targets_series = apply_freq_reduce_mask(x, masks)

                self._optimizer.zero_grad()

                ori_embed = self._feature_fusion(
                    self._encoder(self._input_embedding(x))
                )
                sub_embed = self._subspace_mapper(ori_embed)
                mask_embed = self._mask_encode(masks)

                emb_loss = torch.tensor(0.0, device=self.device)
                mask_loss = torch.tensor(0.0, device=self.device)

                for t in range(self.target_num):
                    target_x = targets_series[:, t]

                    with torch.no_grad():
                        target_embed = self._feature_fusion(
                            self._mom_encoder(
                                self._mom_input_embedding(target_x)
                            )
                        )
                        target_sub = self._mom_subspace_mapper(target_embed)

                    pred_embed = self._embedding_predictor(
                        sub_embed + mask_embed[:, t, :].detach()
                    )
                    mask_pred = self._mask_predictor(
                        target_sub - sub_embed.detach()
                    )

                    emb_loss = emb_loss + criterion(
                        pred_embed, target_sub.detach()
                    )
                    # Upstream (pretrain_models.py) keeps mask_embed live here
                    # (mask_label is NOT detached) so gradient flows into the
                    # learnable mask-embedding parameter (_mask_encode). The
                    # predictor-input side (line above) is detached in upstream too.
                    mask_loss = mask_loss + criterion(
                        mask_pred, mask_embed[:, t, :]
                    )

                loss = emb_loss + mask_loss
                loss.backward()
                self._optimizer.step()

                # EMA update
                ema = self.ema_rate + self._momentum_iter * (
                    1.0 - self.ema_rate
                ) / max(self._total_iters, 1)
                ema = min(ema, 1.0)
                self._momentum_iter += 1
                with torch.no_grad():
                    for p_c, p_m in zip(
                        self._input_embedding.parameters(),
                        self._mom_input_embedding.parameters(),
                    ):
                        p_m.data.mul_(ema).add_((1.0 - ema) * p_c.detach().data)
                    for p_c, p_m in zip(
                        self._encoder.parameters(),
                        self._mom_encoder.parameters(),
                    ):
                        p_m.data.mul_(ema).add_((1.0 - ema) * p_c.detach().data)
                    for p_c, p_m in zip(
                        self._subspace_mapper.parameters(),
                        self._mom_subspace_mapper.parameters(),
                    ):
                        p_m.data.mul_(ema).add_((1.0 - ema) * p_c.detach().data)

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
        raw : bool, if True return (N, T', hidden_dim) without mean pooling

        Returns
        -------
        np.ndarray, shape (N, 128) or (N, T', 128) if raw
        """
        if self._net is None:
            raise RuntimeError(
                "Model has not been trained yet. Call fit() first."
            )

        data = np.nan_to_num(data, nan=0.0).astype(np.float32)
        data_t = torch.from_numpy(data)
        dataset = TensorDataset(data_t)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        was_ie = self._input_embedding.training
        was_enc = self._encoder.training
        self._input_embedding.eval()
        self._encoder.eval()
        outputs = []
        try:
            with torch.no_grad():
                for (batch_data,) in loader:
                    x = batch_data.to(self.device)  # (B, T, D)
                    out = self._encoder(self._input_embedding(x))  # (B, T', hidden_dim)
                    if not raw:
                        out = self._feature_fusion(out)  # (B, hidden_dim)
                    outputs.append(out.cpu().numpy())
        finally:
            if was_ie:
                self._input_embedding.train()
            if was_enc:
                self._encoder.train()

        return np.concatenate(outputs, axis=0)

    # ------------------------------------------------------------------
    @property
    def model(self):
        return self._net
