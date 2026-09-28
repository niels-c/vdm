"""The Allen Brain Observatory cache manifest every BrainObservatoryCache in this repo is built
from.

allensdk resolves a relative manifest path against the process's working directory, so
`python RSM/foo.py` from the repo root and `python foo.py` from RSM/ would each look in a
different cache. Anchoring on config.paths.ALLEN_CACHE_DIR, which is absolute, removes that.
Point ALLEN_BRAIN_OBSERVATORY_CACHE somewhere else (e.g. a scratch disk with more room) to
relocate it; allensdk creates the directory on first use.
"""

import os

from config.paths import ALLEN_CACHE_DIR

ALLEN_MANIFEST_FILE = os.path.join(ALLEN_CACHE_DIR, "manifest.json")

__all__ = ["ALLEN_CACHE_DIR", "ALLEN_MANIFEST_FILE"]
