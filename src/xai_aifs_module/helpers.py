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


def era5_source(keep_cache=False, stale_tmp_hours=24):
    """ARCO ERA5 data source with a bounded disk cache.

    With ``keep_cache=False`` (default) earth2studio downloads into a temporary
    folder that it deletes after every fetch, so initial states are not stored
    permanently (they are rarely reused and each is a full global state).
    Temporary folders left behind by jobs that crashed or hit their time limit
    are removed once older than ``stale_tmp_hours``.
    """
    import shutil
    import time
    from pathlib import Path

    from earth2studio.data import ARCO
    from earth2studio.data.utils import datasource_cache_root

    root = Path(datasource_cache_root()) / "arco_era5"
    cutoff = time.time() - stale_tmp_hours * 3600
    for tmp in root.glob("tmp_arco_era5_*"):
        try:
            if tmp.stat().st_mtime < cutoff:
                shutil.rmtree(tmp, ignore_errors=True)
        except FileNotFoundError:
            pass
    return ARCO(cache=keep_cache)
