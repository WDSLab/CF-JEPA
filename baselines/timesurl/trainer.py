"""TimesURL trainer wrapper."""

import os
import warnings
import numpy as np
from argparse import Namespace
from baselines._import_helper import isolated_import

warnings.filterwarnings("ignore", message=".*torch.cuda.*DtypeTensor.*")

_REPO_DIR = os.path.join(os.path.dirname(__file__), "TimesURL", "src")
_mod = isolated_import(_REPO_DIR, "timesurl")
TimesURL = _mod.TimesURL


class TimesURLTrainer:

    def __init__(self, input_dims, output_dims=320, hidden_dims=64, depth=10,
                 device='cuda', lr=0.0001, batch_size=16,
                 max_train_length=None, **kwargs):
        self._epoch_counter = 0
        self._callback = None
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.max_train_length = max_train_length

        args = Namespace(
            lmd=kwargs.get("lmd", 0.01),
            batch_size=batch_size,
            mask_ratio_per_seg=kwargs.get("mask_ratio_per_seg", 0.05),
            segment_num=kwargs.get("segment_num", 3),
        )

        self._model = TimesURL(
            input_dims=input_dims,  # encoder internally strips the time channel
            output_dims=output_dims, hidden_dims=hidden_dims,
            depth=depth, device=device, lr=lr, batch_size=batch_size,
            max_train_length=max_train_length,
            temporal_unit=kwargs.get("temporal_unit", 0),
            after_epoch_callback=self._epoch_callback_bridge,
            args=args,
        )

    @staticmethod
    def _prepare_train_data(data):
        mask = (~np.isnan(data)).astype(np.float32)
        data_clean = np.nan_to_num(data, nan=0.0)
        N, T, D = data_clean.shape
        time_pos = np.linspace(0, 1, T, dtype=np.float32)[None, :, None]
        time_pos = np.broadcast_to(time_pos, (N, T, 1)).copy()
        x = np.concatenate([data_clean, time_pos], axis=-1)
        return {"x": x, "mask": mask}

    def _epoch_callback_bridge(self, _timesurl_self, cum_loss):
        self._epoch_counter += 1
        if self._callback is not None:
            self._callback(self, self._epoch_counter, cum_loss)

    def fit(self, train_data, epochs, verbose=False, callback=None):
        self._epoch_counter = 0
        self._callback = callback
        prepared = self._prepare_train_data(train_data)
        # Match upstream train.py:145 — paper uses AdamW with constant lr=1e-4.
        # The fit() signature default `is_scheduler=True` is a fallback; the
        # actual paper-faithful call sets it to False unless --sgd is enabled.
        # Without this, CosineAnnealingLR(T_max=epochs) drives lr → 0 by the
        # final epochs, sharply reducing late-stage learning.
        self._model.fit(prepared, n_epochs=self._model.n_epochs + epochs,
                        verbose=verbose, is_scheduler=False)
        return self

    def encode(self, data, raw=False):
        # TimesURL encoder non-dict path splits input as:
        #   x = input[..., :half],  mask = input[..., half:]
        # then from x: time = x[..., -1], features = x[..., :-1]
        # So we need to concat [data, time, mask] → shape (N, T, D+1+D)
        mask = (~np.isnan(data)).astype(np.float32)
        data_clean = np.nan_to_num(data, nan=0.0).astype(np.float32)
        N, T, D = data_clean.shape
        time_pos = np.linspace(0, 1, T, dtype=np.float32)[None, :, None]
        time_pos = np.broadcast_to(time_pos, (N, T, 1)).copy()
        # concat: [data(D), time(1), mask(D)] → (N, T, 2D+1)
        combined = np.concatenate([data_clean, time_pos, mask], axis=-1)
        if raw:
            return self._model.encode(combined, encoding_window=None)
        return self._model.encode(combined, encoding_window="full_series")

    @property
    def device(self):
        return self._model.device

    @property
    def model(self):
        return self._model
