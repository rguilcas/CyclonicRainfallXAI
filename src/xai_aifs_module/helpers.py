import contextlib

import aiohttp
import torch


def fix_aiohttp():
    """Make aiohttp sessions trust proxy environment variables.

    Needed on Olivia to reach ARCO data and the model weights through earth2studio.
    Safe to call more than once.
    """
    if getattr(aiohttp.ClientSession.__init__, "_trust_env_patched", False):
        return
    orig_init = aiohttp.ClientSession.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("trust_env", True)
        orig_init(self, *args, **kwargs)

    patched_init._trust_env_patched = True
    aiohttp.ClientSession.__init__ = patched_init


@contextlib.contextmanager
def grad_through_no_grad():
    """Let gradients flow through code wrapped in torch.no_grad / inference_mode.

    anemoi's predict_step runs under those context managers; inside this block
    they become no-ops. The originals are always restored on exit.
    """
    saved = {cls: (cls.__enter__, cls.__exit__) for cls in (torch.no_grad, torch.inference_mode)}
    try:
        for cls in saved:
            cls.__enter__ = lambda self: None
            cls.__exit__ = lambda self, *args: None
        with torch.enable_grad():
            yield
    finally:
        for cls, (enter, exit_) in saved.items():
            cls.__enter__, cls.__exit__ = enter, exit_
