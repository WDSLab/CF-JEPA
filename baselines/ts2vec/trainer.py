"""TS2Vec trainer wrapper."""

import os
from baselines._import_helper import isolated_import

_REPO_DIR = os.path.join(os.path.dirname(__file__), "ts2vec")
_mod = isolated_import(_REPO_DIR, "ts2vec")
TS2Vec = _mod.TS2Vec


class TS2VecTrainer:

    def __init__(self, input_dims, output_dims=320, hidden_dims=64, depth=10,
                 device='cuda', lr=0.001, batch_size=16,
                 max_train_length=None, **kwargs):
        self._epoch_counter = 0
        self._callback = None
        self.output_dims = output_dims
        self.max_train_length = max_train_length

        self._model = TS2Vec(
            input_dims=input_dims, output_dims=output_dims,
            hidden_dims=hidden_dims, depth=depth, device=device,
            lr=lr, batch_size=batch_size,
            max_train_length=max_train_length,
            temporal_unit=kwargs.get("temporal_unit", 0),
            after_epoch_callback=self._epoch_callback_bridge,
        )

    def _epoch_callback_bridge(self, _ts2vec_self, cum_loss):
        self._epoch_counter += 1
        if self._callback is not None:
            self._callback(self, self._epoch_counter, cum_loss)

    def fit(self, train_data, epochs, verbose=False, callback=None):
        self._epoch_counter = 0
        self._callback = callback
        self._model.fit(train_data, n_epochs=self._model.n_epochs + epochs, verbose=verbose)
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
