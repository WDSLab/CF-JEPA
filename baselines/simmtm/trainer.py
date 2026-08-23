"""
SimMTM trainer wrapper (classification only).

Wraps the TFC model from SimMTM_Classification to expose the unified
cf_jepa trainer interface.  Pre-trains with masked time-series
contrastive + reconstruction losses, then encodes via the 128-dim dense head.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace
from torch.utils.data import TensorDataset, DataLoader

from baselines._import_helper import isolated_import

# ---------------------------------------------------------------------------
# Import original SimMTM components
# ---------------------------------------------------------------------------
_ORIG = os.path.join(
    os.path.dirname(__file__),
    "SimMTM", "SimMTM_Classification", "code",
)
_model_mod = isolated_import(_ORIG, "model")
_loss_mod = isolated_import(_ORIG, "loss")
_aug_mod = isolated_import(_ORIG, "utils.augmentations")

TFC = _model_mod.TFC
ContrastiveWeight = _loss_mod.ContrastiveWeight
AggregationRebuild = _loss_mod.AggregationRebuild
AutomaticWeightedLoss = _loss_mod.AutomaticWeightedLoss
data_transform_masked4cl = _aug_mod.data_transform_masked4cl


# ---------------------------------------------------------------------------
# Helper: compute CNN output temporal length via a dummy forward pass
# ---------------------------------------------------------------------------
def _get_cnn_out_len(input_channels, T, final_out_channels=128,
                     kernel_size=25, stride=3, dropout=0.2):
    """Run a dummy tensor through the 3 conv blocks and return the final
    temporal length (``CNNoutput_channel`` in the original code)."""
    block1 = nn.Sequential(
        nn.Conv1d(input_channels, 32, kernel_size=kernel_size,
                  stride=stride, bias=False, padding=kernel_size // 2),
        nn.BatchNorm1d(32), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        nn.Dropout(dropout),
    )
    block2 = nn.Sequential(
        nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
        nn.BatchNorm1d(64), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
    )
    block3 = nn.Sequential(
        nn.Conv1d(64, final_out_channels, kernel_size=8, stride=1,
                  bias=False, padding=4),
        nn.BatchNorm1d(final_out_channels), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
    )
    with torch.no_grad():
        x = torch.randn(1, input_channels, T)
        x = block1(x)
        x = block2(x)
        x = block3(x)
    return x.shape[2]


class SimMTMTrainer:
    """Unified trainer wrapper around SimMTM (classification pre-training)."""

    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device="cuda",
        lr=0.0001,
        batch_size=16,
        max_train_length=None,
        **kwargs,
    ):
        self.input_dims = input_dims
        self.output_dims = 128  # SimMTM dense head produces 128-dim
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length

        # SimMTM-specific hyper-parameters
        self.kernel_size = kwargs.get("kernel_size", 25)
        self.stride = kwargs.get("stride", 3)
        self.final_out_channels = kwargs.get("final_out_channels", 128)
        self.dropout = kwargs.get("dropout", 0.2)
        self.temperature = kwargs.get("temperature", 0.2)
        self.positive_nums = kwargs.get("positive_nums", 3)
        self.masking_ratio = kwargs.get("masking_ratio", 0.5)
        self.lm = kwargs.get("lm", 3)

        # Model is built lazily in fit() once we know T
        self._net = None
        self._T = None

    # ------------------------------------------------------------------
    def _build_model(self, T):
        """Construct the TFC model with correct dimension configs."""
        cnn_out_len = _get_cnn_out_len(
            self.input_dims, T,
            final_out_channels=self.final_out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dropout=self.dropout,
        )

        configs = SimpleNamespace(
            input_channels=self.input_dims,
            kernel_size=self.kernel_size,
            stride=self.stride,
            final_out_channels=self.final_out_channels,
            dropout=self.dropout,
            CNNoutput_channel=cnn_out_len,
            num_classes_target=2,       # dummy — not used in pretraining
        )
        args = SimpleNamespace(
            training_mode="pre_train",
            temperature=self.temperature,
            positive_nums=self.positive_nums,
        )

        self._net = TFC(configs, args).to(self.device)

        # Fix the reconstruction head to correct dimensions:
        #   input:  CNNoutput_channel * final_out_channels
        #   output: input_channels * T  (reconstruct flattened input)
        head_in = cnn_out_len * self.final_out_channels
        head_out = self.input_dims * T
        self._net.head = nn.Linear(head_in, head_out).to(self.device)
        self._T = T

    # ------------------------------------------------------------------
    def fit(self, train_data, epochs, verbose=False, callback=None):
        """Pre-train the SimMTM encoder.

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

        # Crop to max_train_length if needed (reconstruction head scales with T)
        if self.max_train_length is not None and T > self.max_train_length:
            data = data[:, :self.max_train_length, :]
            T = self.max_train_length

        # Build / rebuild model if T changed
        if self._net is None or self._T != T:
            self._build_model(T)
            self._optimizer = None
            self._scheduler = None
            self._epoch_counter = 0

        # (N, T, D) -> (N, D, T) for Conv1d
        data_t = torch.from_numpy(data).permute(0, 2, 1)
        dataset = TensorDataset(data_t)
        # drop_last=False: when N < batch_size (e.g. UEA AtrialFibrillation N=15,
        # batch=16), drop_last=True would yield 0 batches and silently skip all
        # training — encoder would stay at random init with no error raised.
        loader = DataLoader(
            dataset, batch_size=min(self.batch_size, N),
            shuffle=True, drop_last=False,
        )

        # Create optimizer/scheduler only once
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=0)
            self._total_epochs = epochs
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=epochs,
            )
        else:
            # Incremental fit: update T_max to reflect total training so far
            new_total = self._epoch_counter + epochs
            if new_total > self._total_epochs:
                self._total_epochs = new_total
                self._scheduler.T_max = new_total

        for epoch in range(1, epochs + 1):
            self._epoch_counter += 1
            self._net.train()
            cum_loss = 0.0
            n_iters = 0

            for (batch_data,) in loader:
                data_masked, _mask = data_transform_masked4cl(
                    batch_data, self.masking_ratio, self.lm,
                    self.positive_nums,
                )
                data_combined = torch.cat(
                    [batch_data, data_masked], dim=0,
                ).to(self.device)

                self._optimizer.zero_grad()
                loss, _loss_cl, _loss_rb = self._net(
                    data_combined, pretrain=True,
                )
                loss.backward()
                self._optimizer.step()

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
        """Produce fixed-length representations.

        Parameters
        ----------
        data : np.ndarray, shape (N, T, D)
        raw : bool, ignored — SimMTM always returns (N, 128) due to architecture

        Returns
        -------
        np.ndarray, shape (N, 128)
        """
        if self._net is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        data = np.nan_to_num(data, nan=0.0).astype(np.float32)
        # Crop to model's expected T (must match training)
        if self._T is not None and data.shape[1] > self._T:
            data = data[:, :self._T, :]
        data_t = torch.from_numpy(data).permute(0, 2, 1)  # (N, D, T)
        dataset = TensorDataset(data_t)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        was_training = self._net.training
        self._net.eval()
        outputs = []
        try:
            with torch.no_grad():
                for (batch_data,) in loader:
                    x = batch_data.to(self.device)
                    _h, z = self._net(x, pretrain=False)
                    outputs.append(z.cpu().numpy())
        finally:
            if was_training:
                self._net.train()

        return np.concatenate(outputs, axis=0)

    # ------------------------------------------------------------------
    @property
    def model(self):
        return self._net
