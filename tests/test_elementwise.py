"""Correctness tests for elementwise operators."""

import ctypes

import ml_dtypes
import numpy as np
import pytest

from tests.helpers import GuardedOutput, bind_launch


@pytest.fixture(scope="module")
def add():
    return bind_launch("elementwise/add", "launch_add", [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_void_p,
    ])


def _check_add(hip, add, a, b):
    with (
        hip.from_numpy(a) as device_a,
        hip.from_numpy(b) as device_b,
        GuardedOutput(hip, a.shape, a.dtype) as output,
    ):
        hip.check(add(device_a.ptr, device_b.ptr, output.ptr, a.size, None))
        actual = output.read()

    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.add(a.astype(np.float32), b.astype(np.float32)).astype(
            ml_dtypes.bfloat16,
        )
    # BF16 值可精确转换为 FP32，便于 NumPy 正确比较 NaN 等特殊值。
    np.testing.assert_array_equal(
        actual.astype(np.float32), expected.astype(np.float32),
    )


@pytest.mark.parametrize("n", [
    0, 1, 31, 32, 33, 255, 256, 257, 1025, 1 << 20, 1 << 24,
])
def test_add(hip, add, n):
    rng = np.random.default_rng(42)
    a = rng.standard_normal(n, dtype=np.float32).astype(ml_dtypes.bfloat16)
    b = rng.standard_normal(n, dtype=np.float32).astype(ml_dtypes.bfloat16)
    _check_add(hip, add, a, b)


def test_add_special_values(hip, add):
    limit = ml_dtypes.finfo(ml_dtypes.bfloat16)
    a = np.array([
        0.0, -0.0, 1.0, -1.0, limit.max, limit.tiny,
        np.inf, -np.inf, np.inf, np.nan, 2**8,
        1.0, 1.0 + 2**-7,
    ], dtype=ml_dtypes.bfloat16)
    b = np.array([
        0.0, -0.0, -1.0, 1.0, limit.max, limit.tiny,
        1.0, -1.0, -np.inf, 1.0, 1.0,
        2**-8, 2**-8,
    ], dtype=ml_dtypes.bfloat16)
    _check_add(hip, add, a, b)
