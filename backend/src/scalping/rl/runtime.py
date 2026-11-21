"""Thread pinning for the 8 GB target machine.

Importing this module does not change process state; :func:`pin_threads` is
called from :func:`~scalping.rl.train.train` and
:func:`~scalping.rl.policy.load_policy` so a short inference call is not stuck
with OpenMP's default worker pool either.
"""

from __future__ import annotations

import os

import torch

__all__ = ["pin_threads"]


def pin_threads() -> None:
    """Pin torch and its BLAS pools to one thread.

    The environment variables are exported for libraries that spawn workers
    later; ``torch.set_num_threads`` is what bounds the current process.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
