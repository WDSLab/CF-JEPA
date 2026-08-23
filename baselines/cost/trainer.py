"""CoST trainer wrapper."""

import os
import numpy as np
from baselines._import_helper import isolated_import

_REPO_DIR = os.path.join(os.path.dirname(__file__), "CoST")
_mod = isolated_import(_REPO_DIR, "cost")
CoST = _mod.CoST


class CoSTTrainer:

    def __init__(self, input_dims, output_dims=320, hidden_dims=64, depth=10,
                 device='cuda', lr=0.001, batch_size=16,
                 max_train_length=3000, **kwargs):
        self._epoch_counter = 0
        self._callback = None
        self.output_dims = output_dims
        # Paper default: 3000 (matches train.py:35 CLI default). Long sequences
        # are chunked into <=3000-length sections during fit, then a 3000-length
        # random window is sampled per batch. BandedFourierLayer weights are
        # sized to max_train_length, so this also caps Fourier capacity.
        self.max_train_length = max_train_length
        self._max_train_length = max_train_length  # legacy alias used in _build_model

        # Save params for lazy model creation in fit() (need T for max_train_length)
        self._init_kwargs = dict(
            input_dims=input_dims, output_dims=output_dims,
            hidden_dims=hidden_dims, depth=depth, device=device,
            lr=lr, batch_size=batch_size,
            kernels=kwargs.get("kernels", [1, 2, 4, 8, 16, 32, 64, 128]),
            alpha=kwargs.get("alpha", 0.0005),
        )
        self._model = None

    def _build_model(self, T):
        # BandedFourierLayer weight is sized to max_train_length // 2 + 1 frequencies.
        # If actual sequence length T < max_train_length, FFT output has fewer freqs
        # than weights expect → einsum mismatch. So cap mtl at T as well as the user
        # value: long datasets get chunked into max_train_length pieces, short
        # datasets get weights sized to their actual length.
        if self._max_train_length is None:
            mtl = T
        else:
            mtl = min(self._max_train_length, T)
        self._model = CoST(
            **self._init_kwargs,
            max_train_length=mtl,
            after_epoch_callback=self._epoch_callback_bridge,
        )

    def _epoch_callback_bridge(self, _cost_self, cum_loss):
        self._epoch_counter += 1
        if self._callback is not None:
            self._callback(self, self._epoch_counter, cum_loss)

    def fit(self, train_data, epochs, verbose=False, callback=None):
        data_clean = np.nan_to_num(train_data, nan=0.0).astype(np.float32)
        T = data_clean.shape[1]

        if self._model is None:
            self._build_model(T)

        self._epoch_counter = 0
        self._callback = callback
        self._model.fit(data_clean, n_epochs=self._model.n_epochs + epochs, verbose=verbose)
        return self

    def encode(self, data, raw=False):
        data_clean = np.nan_to_num(data, nan=0.0).astype(np.float32)
        if raw:
            return self._encode_raw(data_clean)
        out = self._model.encode(data_clean, mode="forecasting")
        if out.ndim == 3:
            out = out.squeeze(1)
        return out

    def _encode_raw(self, data):
        """Return (N, T, D) pre-pooling representations.
        CoST encoder outputs trend (B,T,D/2) and season (B,T,D/2),
        concatenated along feature dim → (B, T, D).

        CoST's BandedFourierLayer requires input length == max_train_length.
        Inputs are right-padded to max_train_length, then trimmed back.
        Long inputs are split into max_train_length chunks first.
        """
        import torch
        mtl = self._model.max_train_length
        N, T, C = data.shape

        # Split long sequences into chunks
        if mtl is not None and T > mtl:
            chunk_specs = []
            for start in range(0, T, mtl):
                end = min(start + mtl, T)
                chunk_specs.append((start, end))
        else:
            chunk_specs = [(0, T)]

        # CRITICAL: self._model.net is shared with self._model.cost (training module
        # holds self.net as encoder_q at cost.py:234). Setting eval here without
        # restoring causes CoST training (cost.py:286-322 has no .train() call) to
        # continue in eval mode after first mid-training encode. Save/restore is
        # required under the continuous-fit pipeline.
        was_training = self._model.net.training
        self._model.net.eval()
        encode_batch = getattr(self._model, 'batch_size', 64) or 64
        all_chunk_outputs = []
        try:
            with torch.no_grad():
                for start, end in chunk_specs:
                    chunk = data[:, start:end, :]
                    actual_len = chunk.shape[1]
                    # Pad to exactly max_train_length (FFT weight requirement)
                    if mtl is not None and actual_len < mtl:
                        pad = np.zeros((N, mtl - actual_len, C), dtype=chunk.dtype)
                        chunk = np.concatenate([chunk, pad], axis=1)
                    # Encode in batches to avoid OOM
                    batch_outs = []
                    for i in range(0, N, encode_batch):
                        x = torch.from_numpy(chunk[i:i+encode_batch]).float().to(self._model.device)
                        trend, season = self._model.net(x)
                        out = torch.cat([trend, season], dim=-1)
                        batch_outs.append(out[:, :actual_len].cpu().numpy())
                    all_chunk_outputs.append(np.concatenate(batch_outs, axis=0))
        finally:
            if was_training:
                self._model.net.train()

        return np.concatenate(all_chunk_outputs, axis=1)  # (N, T, D)

    @property
    def device(self):
        return self._model.device

    @property
    def model(self):
        return self._model
