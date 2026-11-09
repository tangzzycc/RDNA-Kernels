"""Correctness tests for matrix multiplication launch entry points."""

import ctypes

import numpy as np
import pytest

from tests.helpers import GuardedOutput, bind_launch


VERSIONS = [1, 2, 3, 4, 5, 6, 9, 10]


@pytest.fixture(scope="module")
def sgemm():
    return bind_launch("matmul/sgemm", "launch_sgemm", [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ])


def _check_sgemm(hip, sgemm, version, shape, stream=None):
    M, N, K = shape
    rng = np.random.default_rng(2026)
    A = rng.uniform(-0.5, 0.5, (M, K)).astype(np.float32)
    B = rng.uniform(-0.5, 0.5, (K, N)).astype(np.float32)
    with (
        hip.from_numpy(A) as device_A,
        hip.from_numpy(B) as device_B,
        GuardedOutput(hip, (M, N)) as output,
    ):
        hip.check(sgemm(device_A.ptr, device_B.ptr, output.ptr,
                       M, N, K, version, stream))
        actual = output.read()
        after_A, after_B = np.empty_like(A), np.empty_like(B)
        device_A.copy_to(after_A)
        device_B.copy_to(after_B)
        np.testing.assert_array_equal(after_A, A)
        np.testing.assert_array_equal(after_B, B)
    expected = A.astype(np.float64) @ B.astype(np.float64)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("shape", [(128, 256, 32), (256, 256, 64)])
def test_sgemm(hip, sgemm, version, shape):
    _check_sgemm(hip, sgemm, version, shape)


@pytest.mark.parametrize("version", [1, 2])
def test_sgemm_partial_tiles(hip, sgemm, version):
    _check_sgemm(hip, sgemm, version, (67, 93, 19))


@pytest.mark.parametrize("version", VERSIONS)
def test_sgemm_nondefault_stream(hip, sgemm, hip_stream, version):
    _check_sgemm(hip, sgemm, version, (128, 256, 64), hip_stream)


@pytest.mark.parametrize("version", VERSIONS)
def test_sgemm_empty(sgemm, version):
    assert sgemm(None, None, None, 0, 256, 32, version, None) == 0


@pytest.mark.parametrize("version", [0, 7, 8, 11])
def test_sgemm_unknown_version(sgemm, version):
    assert sgemm(None, None, None, 128, 256, 32, version, None) != 0


@pytest.mark.parametrize("version", [3, 4, 5, 6, 9, 10])
def test_sgemm_partial_tile_rejected(hip, sgemm, version):
    with hip.buffer(16) as data:
        assert sgemm(data.ptr, data.ptr, data.ptr, 127, 256, 32, version, None) != 0
