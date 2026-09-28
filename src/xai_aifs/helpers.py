import aiohttp
import torch

def fix_aiohttp():
    """
    Patch aiohttp to trust environment variables for proxy settings.
    This is required for accessing some of ARCO data and model weights with nvidia earth2studio on Olivia.
    """
    _orig_init = aiohttp.ClientSession.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("trust_env", True)
        _orig_init(self, *args, **kwargs)
    aiohttp.ClientSession.__init__ = _patched_init


# ----------------------------------------------------------------------------
# Gradients through anemoi's internal no_grad / inference_mode
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def grad_through_no_grad():
    """anemoi's predict_step runs inside torch.no_grad / inference_mode.
    Make those context managers no-ops only while this block is active."""
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
