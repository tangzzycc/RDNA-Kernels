"""Reusable launch bindings and output boundary checks."""

import ctypes
from math import prod

import numpy as np

from common.python import load_kernel


def bind_launch(operator: str, symbol: str, argtypes: list):
    """Bind a host launch function returning hipError_t."""
    library = load_kernel(operator)
    launch = getattr(library, symbol)
    launch.argtypes = argtypes
    launch.restype = ctypes.c_int
    return launch


class GuardedOutput:
    """Output allocation with sentinels before and after the logical tensor."""

    def __init__(self, hip, shape, dtype=np.float32):
        self.hip = hip
        self.shape = tuple(shape)
        self.guard = 256
        self.sentinel = np.asarray(85, dtype=dtype).item()
        self.host = np.full(prod(self.shape) + 2 * self.guard, self.sentinel, dtype=dtype)
        self.device = hip.from_numpy(self.host)
        self.ptr = ctypes.c_void_p(self.device.ptr.value + self.guard * self.host.itemsize)

    def read(self) -> np.ndarray:
        """Synchronize, check guards, and return the output for numerical checks."""
        self.hip.synchronize()
        self.device.copy_to(self.host)
        np.testing.assert_array_equal(self.host[:self.guard], self.sentinel)
        np.testing.assert_array_equal(self.host[-self.guard:], self.sentinel)
        return self.host[self.guard:-self.guard].reshape(self.shape)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.device.close()
