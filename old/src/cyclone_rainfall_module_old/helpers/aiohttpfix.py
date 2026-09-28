import aiohttp

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
