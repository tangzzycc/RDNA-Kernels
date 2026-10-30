"""Minimal HIP runtime bindings and device buffer ownership."""

import ctypes

import numpy as np

from .sdk import sdk_root


class HipRuntime:
    """Launch functions return HIP status codes, checked with check()."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL(str(sdk_root() / "lib/libamdhip64.so"))
        signatures = {
            "hipMalloc": ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
            "hipFree": ([ctypes.c_void_p], ctypes.c_int),
            "hipMemcpy": (
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
                ctypes.c_int,
            ),
            "hipDeviceSynchronize": ([], ctypes.c_int),
            "hipGetErrorString": ([ctypes.c_int], ctypes.c_char_p),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = argtypes
            function.restype = restype

    def check(self, status: int) -> None:
        if status != 0:
            message = self.lib.hipGetErrorString(status).decode()
            raise RuntimeError(f"HIP error {status}: {message}")

    def synchronize(self) -> None:
        self.check(self.lib.hipDeviceSynchronize())

    def buffer(self, nbytes: int) -> "DeviceBuffer":
        return DeviceBuffer(self, nbytes)

    def from_numpy(self, array: np.ndarray) -> "DeviceBuffer":
        buffer = self.buffer(array.nbytes)
        try:
            buffer.copy_from(array)
        except Exception:
            buffer.close()
            raise
        return buffer


class DeviceBuffer:
    """A device allocation used as a context manager, including empty arrays."""

    def __init__(self, hip: HipRuntime, nbytes: int) -> None:
        if nbytes < 0:
            raise ValueError("Buffer size must be nonnegative")
        self.hip = hip
        self.nbytes = nbytes
        self.ptr = ctypes.c_void_p()
        # Keep a valid pointer for zero-length inputs without launching work.
        hip.check(hip.lib.hipMalloc(ctypes.byref(self.ptr), max(nbytes, 1)))

    def _copy(self, array: np.ndarray, to_device: bool) -> None:
        if self.ptr.value is None:
            raise RuntimeError("Device buffer is closed")
        if not array.flags.c_contiguous or array.nbytes > self.nbytes:
            raise ValueError("Copy requires a contiguous array that fits in the buffer")
        if not to_device and not array.flags.writeable:
            raise ValueError("Copy destination must be writable")
        if array.nbytes:
            host_ptr = ctypes.c_void_p(array.ctypes.data)
            destination, source = (self.ptr, host_ptr) if to_device else (host_ptr, self.ptr)
            # hipMemcpyHostToDevice = 1; hipMemcpyDeviceToHost = 2.
            self.hip.check(self.hip.lib.hipMemcpy(
                destination, source, array.nbytes, 1 if to_device else 2,
            ))

    def copy_from(self, array: np.ndarray) -> None:
        self._copy(array, to_device=True)

    def copy_to(self, array: np.ndarray) -> None:
        self._copy(array, to_device=False)

    def close(self) -> None:
        if self.ptr.value is not None:
            self.hip.check(self.hip.lib.hipFree(self.ptr))
            self.ptr = ctypes.c_void_p()

    def __enter__(self) -> "DeviceBuffer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
