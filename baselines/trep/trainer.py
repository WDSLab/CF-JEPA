"""T-Rep trainer wrapper."""

import os
import numpy as np
from baselines._import_helper import isolated_import
from baselines._epoch_hook import install_epoch_hook

_REPO_DIR = os.path.join(os.path.dirname(__file__), "T-Rep")
_mod = isolated_import(_REPO_DIR, "trep")
TRep = _mod.TRep


class TRepTrainer:

    def __init__(self, input_dims, output_dims=128, hidden_dims=None, depth=10,
                 device='cuda', lr=0.001, batch_size=16,
                 max_train_length=None, **kwargs):
        # T-Rep paper §5.2 explicitly designs around F=128 (vs 320 baselines):
        # "T-Rep outperforms all baselines with a latent space that is almost
        # 3 times smaller, using only F = 128 dimensions ... attributed to its
        # comprehensive treatment of time ... with its time-embedding module."
        # Following paper-faithful policy (parallel to architecture-fixed
        # exceptions SimMTM 128, TS-JEPA 384, FEI 1024), we keep F=128 — the
        # method's claimed contribution depends on smaller-dim sufficiency.
        self.output_dims = output_dims
        self.max_train_length = max_train_length
        # Use original T-Rep paper's layer-wise hidden dims
        if hidden_dims is None:
            hidden_dims = np.array([64, 128, 256, 256, 256, 256, 128, 128, 128, 128])

        self._model = TRep(
            input_dims=input_dims, output_dims=output_dims,
            hidden_dims=hidden_dims, depth=depth, device=device,
            lr=lr, batch_size=batch_size,
            time_embedding=kwargs.get("time_embedding", "t2v_sin"),
            time_embedding_dim=kwargs.get("time_embedding_dim", 64),
            task_weights=kwargs.get("task_weights"),
            max_train_length=max_train_length,
            temporal_unit=kwargs.get("temporal_unit", 0),
        )

        install_epoch_hook(self._model)
        self._callback = None

        def _hook(model, ep, loss):
            if self._callback is not None:
                self._callback(self, ep, loss)
        self._model._after_epoch_callback = _hook

    def fit(self, train_data, epochs, verbose=False, callback=None):
        # Single continuous fit() call; per-epoch callback fires via the
        # n_epochs __setattr__ hook installed in __init__.
        self._callback = callback
        target = self._model.n_epochs + epochs
        self._model.fit(train_data, n_epochs=target, verbose=verbose)
        self._callback = None
        return self

    def encode(self, data, raw=False):
        if raw:
            return self._model.encode(data, encoding_window=None)
        return self._model.encode(data, encoding_window="full_series")

    @property
    def device(self):
        return self._model.device

    @property
    def model(self):
        return self._model
