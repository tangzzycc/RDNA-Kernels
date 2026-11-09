"""提供 HIP 运行时调用与设备缓冲区管理."""

import ctypes

import numpy as np

from .sdk import sdk_root


class HipRuntime:
    """封装项目 ROCm SDK 的 HIP 内存操作, 状态检查和设备同步."""

    def __init__(self) -> None:
        """加载项目 SDK 的 HIP 运行时库, 并设置 ctypes 函数签名."""
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
        """检查 HIP 状态码, 失败时抛出包含错误信息的 RuntimeError."""
        if status != 0:
            message = self.lib.hipGetErrorString(status).decode()
            raise RuntimeError(f"HIP error {status}: {message}")

    def synchronize(self) -> None:
        """等待当前设备上所有流中已提交的任务完成."""
        self.check(self.lib.hipDeviceSynchronize())

    def buffer(self, nbytes: int) -> "DeviceBuffer":
        """返回新分配的设备缓冲区, nbytes 为缓冲区的字节容量."""
        return DeviceBuffer(self, nbytes)

    def from_numpy(self, array: np.ndarray) -> "DeviceBuffer":
        """分配设备缓冲区并复制 C 连续的 NumPy 数组, 保留原始字节布局."""
        buffer = self.buffer(array.nbytes)
        try:
            buffer.copy_from(array)
        except Exception:
            buffer.close()
            raise
        return buffer


class DeviceBuffer:
    """管理一块设备内存, 支持使用 with 在作用域结束时释放资源."""

    def __init__(self, hip: HipRuntime, nbytes: int) -> None:
        """分配非负字节容量的设备内存, 零容量也保留一个有效指针."""
        if nbytes < 0:
            raise ValueError("Buffer size must be nonnegative")
        self.hip = hip
        self.nbytes = nbytes
        self.ptr = ctypes.c_void_p()
        # 零容量时实际分配一个字节, 保持指针有效.
        hip.check(hip.lib.hipMalloc(ctypes.byref(self.ptr), max(nbytes, 1)))

    def _copy(self, array: np.ndarray, to_device: bool) -> None:
        """按 to_device 指定的方向复制数组字节, 不转换 dtype.

        数组须为 C 连续且不超过缓冲区容量, 主机侧的目标数组还须可写.
        """
        if self.ptr.value is None:
            raise RuntimeError("Device buffer is closed")
        if not array.flags.c_contiguous or array.nbytes > self.nbytes:
            raise ValueError("Copy requires a contiguous array that fits in the buffer")
        if not to_device and not array.flags.writeable:
            raise ValueError("Copy destination must be writable")
        if array.nbytes:
            host_ptr = ctypes.c_void_p(array.ctypes.data)
            destination, source = (self.ptr, host_ptr) if to_device else (host_ptr, self.ptr)
            # 复制方向: hipMemcpyHostToDevice = 1, hipMemcpyDeviceToHost = 2.
            self.hip.check(self.hip.lib.hipMemcpy(
                destination, source, array.nbytes, 1 if to_device else 2,
            ))

    def copy_from(self, array: np.ndarray) -> None:
        """将 C 连续的 NumPy 数组复制到设备缓冲区的起始位置."""
        self._copy(array, to_device=True)

    def copy_to(self, array: np.ndarray) -> None:
        """读取设备缓冲区起始区域, 复制到可写且 C 连续的 NumPy 数组."""
        self._copy(array, to_device=False)

    def close(self) -> None:
        """释放设备内存并清空指针, 重复调用不会重复释放."""
        if self.ptr.value is not None:
            self.hip.check(self.hip.lib.hipFree(self.ptr))
            self.ptr = ctypes.c_void_p()

    def __enter__(self) -> "DeviceBuffer":
        """返回当前缓冲区, 供 with 语句使用."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """离开 with 作用域时释放设备内存."""
        self.close()
