import ctypes

import pytest

from common.python import HipRuntime


@pytest.fixture(scope="session")
def hip():
    return HipRuntime()


@pytest.fixture
def hip_stream(hip):
    """创建非默认 HIP stream, 测试结束后释放."""
    create = hip.lib.hipStreamCreateWithFlags
    create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    create.restype = ctypes.c_int
    destroy = hip.lib.hipStreamDestroy
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = ctypes.c_int
    stream = ctypes.c_void_p()
    hip.check(create(ctypes.byref(stream), 1))
    try:
        yield stream
    finally:
        hip.check(destroy(stream))
