"""Correctness tests for normalization operators."""

import ctypes

import ml_dtypes
import numpy as np
import pytest

from tests.helpers import GuardedOutput, bind_launch


@pytest.fixture(scope="module")
def rmsnorm():
    return bind_launch("norm/rmsnorm", "launch_rmsnorm", [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
    ])


def _check_rmsnorm(hip, rmsnorm, x, gamma, eps=1e-6, x_offset=0, gamma_offset=0):
    x_storage = np.full(x.size + x_offset, 85, dtype=x.dtype)
    x_storage[x_offset:] = x.ravel()
    gamma_storage = np.full(gamma.size + gamma_offset, 85, dtype=gamma.dtype)
    gamma_storage[gamma_offset:] = gamma
    with (
        hip.from_numpy(x_storage) as device_x,
        hip.from_numpy(gamma_storage) as device_gamma,
        GuardedOutput(hip, x.shape, x.dtype) as output,
    ):
        x_ptr = ctypes.c_void_p(device_x.ptr.value + x_offset * x.itemsize)
        gamma_ptr = ctypes.c_void_p(
            device_gamma.ptr.value + gamma_offset * gamma.itemsize,
        )
        hip.check(rmsnorm(
            x_ptr, gamma_ptr, output.ptr, *x.shape, eps, None,
        ))
        actual = output.read()

    # 使用高精度数学参考；不同 FP32 归约顺序允许最终结果相差一个 BF16 ULP。
    xf = x.astype(np.float64)
    expected = (
        xf / np.sqrt(np.mean(xf * xf, axis=1, keepdims=True) + eps)
        * gamma.astype(np.float64)
    ).astype(ml_dtypes.bfloat16)
    distance = np.abs(
        actual.view(np.uint16).astype(np.int32)
        - expected.view(np.uint16).astype(np.int32),
    )
    distance[actual.astype(np.float32) == expected.astype(np.float32)] = 0
    assert np.max(distance) <= 1, f"Maximum BF16 ULP distance: {np.max(distance)}"


@pytest.mark.parametrize("shape", [
    (1, 1), (1, 128), (8, 130), (8, 132), (8, 513),
    (32, 5120), (255, 5120), (256, 5120), (1024, 5120),
    (256, 4096), (256, 8192),
])
def test_rmsnorm(hip, rmsnorm, shape):
    rng = np.random.default_rng(42)
    x = rng.standard_normal(shape, dtype=np.float32).astype(ml_dtypes.bfloat16)
    gamma = rng.standard_normal(shape[-1], dtype=np.float32).astype(
        ml_dtypes.bfloat16,
    )
    _check_rmsnorm(hip, rmsnorm, x, gamma)


@pytest.mark.parametrize("offsets", [(1, 0), (0, 1), (2, 0), (4, 0)])
def test_rmsnorm_offset_input(hip, rmsnorm, offsets):
    rng = np.random.default_rng(2026)
    x = rng.standard_normal((32, 5120), dtype=np.float32).astype(ml_dtypes.bfloat16)
    gamma = rng.standard_normal(5120, dtype=np.float32).astype(ml_dtypes.bfloat16)
    _check_rmsnorm(hip, rmsnorm, x, gamma, x_offset=offsets[0], gamma_offset=offsets[1])


def test_rmsnorm_zero_input(hip, rmsnorm):
    x = np.zeros((8, 513), dtype=ml_dtypes.bfloat16)
    gamma = np.ones(513, dtype=ml_dtypes.bfloat16)
    _check_rmsnorm(hip, rmsnorm, x, gamma)


@pytest.fixture(scope="module")
def add_rmsnorm():
    return bind_launch("norm/add_rmsnorm", "launch_add_rmsnorm", [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p,
    ])


def _check_add_rmsnorm(hip, add_rmsnorm, shape, stream=None):
    rng = np.random.default_rng(2026)
    x = rng.standard_normal(shape).astype(ml_dtypes.bfloat16)
    residual = rng.standard_normal(shape).astype(ml_dtypes.bfloat16)
    gamma = rng.standard_normal(shape[-1]).astype(ml_dtypes.bfloat16)
    with (
        GuardedOutput(hip, shape, x.dtype) as device_x,
        GuardedOutput(hip, shape, x.dtype) as device_residual,
        hip.from_numpy(gamma) as device_gamma,
    ):
        device_x.host[device_x.guard:-device_x.guard] = x.ravel()
        device_residual.host[device_residual.guard:-device_residual.guard] = residual.ravel()
        device_x.device.copy_from(device_x.host)
        device_residual.device.copy_from(device_residual.host)
        hip.check(add_rmsnorm(device_x.ptr, device_residual.ptr, device_gamma.ptr,
                             *shape, 1e-6, stream))
        actual_x, actual_residual = device_x.read(), device_residual.read()

    expected_residual = (x.astype(np.float32) + residual.astype(np.float32)).astype(
        ml_dtypes.bfloat16,
    )
    np.testing.assert_array_equal(actual_residual, expected_residual)
    summed = expected_residual.astype(np.float64)
    expected_x = (summed / np.sqrt(np.mean(summed * summed, axis=1, keepdims=True) + 1e-6)
                  * gamma.astype(np.float64)).astype(ml_dtypes.bfloat16)
    distance = np.abs(actual_x.view(np.uint16).astype(np.int32)
                      - expected_x.view(np.uint16).astype(np.int32))
    distance[actual_x.astype(np.float32) == expected_x.astype(np.float32)] = 0
    assert np.max(distance) <= 1, f"Maximum BF16 ULP distance: {np.max(distance)}"


@pytest.mark.parametrize("shape", [(1, 128), (8, 513), (1024, 5120)])
def test_add_rmsnorm(hip, add_rmsnorm, shape):
    _check_add_rmsnorm(hip, add_rmsnorm, shape)


def test_add_rmsnorm_nondefault_stream(hip, add_rmsnorm, hip_stream):
    _check_add_rmsnorm(hip, add_rmsnorm, (8, 513), hip_stream)
