"""Shared per-epoch callback hook for upstream models lacking a native hook.

Methods like AutoTCL, T-Rep, InfoTS all increment ``self.n_epochs``
once per epoch in their training loop, but expose no ``after_epoch_callback``
argument. To support our continuous-fit pipeline (single ``fit(max_epoch)``
call + per-epoch callback for checkpoint encode/eval), we install a dynamic
subclass whose ``__setattr__`` intercepts ``n_epochs`` increments and fires
a registered callback.

Loss resolution
---------------
The callback is invoked with the current epoch's training loss. We resolve
the value in this order:

1. ``model._last_epoch_loss`` if upstream explicitly publishes it.
2. Frame walk: search the call stack for a local ``loss_log`` list (the
   pattern all three upstream ``fit()`` methods use) and return its last
   element. T-Rep appends before ``n_epochs += 1`` so this yields the
   current epoch's loss exactly; AutoTCL/InfoTS append after the increment,
   so this yields the previous epoch's loss (one-step lag — still beats
   the constant-0 fallback for CSV logging and loss-curve visualisation).
3. ``0.0`` as a last-resort default.

Usage
-----
    install_epoch_hook(model)
    model._after_epoch_callback = lambda model, epoch, loss: ...
"""

import sys


def install_epoch_hook(model):
    """Install a per-epoch __setattr__ hook on *model*.

    After installation, setting ``model._after_epoch_callback = fn`` makes
    ``fn(model, new_epoch, loss)`` fire every time ``self.n_epochs`` is
    incremented inside upstream's training loop.
    """
    orig_cls = model.__class__
    if getattr(orig_cls, '__epoch_hook_installed', False):
        return

    class _Hooked(orig_cls):
        def __setattr__(self, name, value):
            if name == 'n_epochs':
                prev = self.__dict__.get('n_epochs', 0)
                object.__setattr__(self, name, value)
                hook = self.__dict__.get('_after_epoch_callback')
                if hook is not None and value > prev:
                    hook(self, value, _resolve_epoch_loss(self))
            else:
                object.__setattr__(self, name, value)

    _Hooked.__name__ = orig_cls.__name__ + 'Hooked'
    _Hooked.__epoch_hook_installed = True
    model.__class__ = _Hooked


def _resolve_epoch_loss(model):
    """Return the most recent epoch loss; see module docstring."""
    explicit = model.__dict__.get('_last_epoch_loss')
    if explicit is not None:
        return explicit
    # Walk up the stack: this fn (0) -> __setattr__ (1) -> upstream caller (2+).
    # All three upstream fit() loops maintain a local ``loss_log`` list.
    frame = sys._getframe(2)
    while frame is not None:
        loss_log = frame.f_locals.get('loss_log')
        if isinstance(loss_log, list) and loss_log:
            return loss_log[-1]
        frame = frame.f_back
    return 0.0
