import aiohttp

_orig_init = aiohttp.ClientSession.__init__
def _patched_init(self, *args, **kwargs):
    kwargs.setdefault("trust_env", True)
    _orig_init(self, *args, **kwargs)
aiohttp.ClientSession.__init__ = _patched_init



import os
os.makedirs("outputs", exist_ok=True)

from earth2studio.data import IFS, GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import AIFS, SFNO
import earth2studio.run as run

package = AIFS.load_default_package()   # or load_default_package(version="1.1")
model = AIFS.load_model(package)

day = "2006-07-08"
data = GFS()
io = ZarrBackend(file_name=f"outputs/aifs_forecast_{day}.zarr")

nsteps = 20  # 20 steps x 6h = 5 days
io = run.deterministic([day], nsteps, model, data, io)
print(io.root.tree())