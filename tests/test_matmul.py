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


@pytest.fixture(scope="module", params=[np.float32, np.float16], ids=["fp32", "fp16"])
def opt_gemm(request):
    dtype = request.param
    name = "sgemm_opt_fp16" if dtype == np.float16 else "sgemm_opt"
    launch = bind_launch(f"matmul/{name}", f"launch_{name}", [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ])
    return dtype, launch


def _check_gemm(hip, launch, version, shape, stream=None, dtype=np.float32):
    M, N, K = shape
    rng = np.random.default_rng(2026)
    A = rng.uniform(-0.5, 0.5, (M, K)).astype(dtype)
    B = rng.uniform(-0.5, 0.5, (K, N)).astype(dtype)
    with (
        hip.from_numpy(A) as device_A,
        hip.from_numpy(B) as device_B,
        GuardedOutput(hip, (M, N), dtype=dtype) as output,
    ):
        args = [device_A.ptr, device_B.ptr, output.ptr, M, N, K]
        if version is not None:
            args.append(version)
        hip.check(launch(*args, stream))
        actual = output.read()
        after_A, after_B = np.empty_like(A), np.empty_like(B)
        device_A.copy_to(after_A)
        device_B.copy_to(after_B)
        np.testing.assert_array_equal(after_A, A)
        np.testing.assert_array_equal(after_B, B)
    expected = A.astype(np.float64) @ B.astype(np.float64)
    rtol, atol = (6e-4, 2e-5) if dtype == np.float16 else (2e-5, 2e-6)
    if dtype == np.float32:
        # 长 K 的逐项累加误差增大, 为接近零的输出按 K 扩展绝对容差.
        atol *= max(1, K / 128)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("shape", [(128, 256, 32), (256, 256, 64)])
def test_sgemm(hip, sgemm, version, shape):
    _check_gemm(hip, sgemm, version, shape)


@pytest.mark.parametrize("version", [1, 2])
def test_sgemm_partial_tiles(hip, sgemm, version):
    _check_gemm(hip, sgemm, version, (67, 93, 19))


@pytest.mark.parametrize("version", VERSIONS)
def test_sgemm_nondefault_stream(hip, sgemm, hip_stream, version):
    _check_gemm(hip, sgemm, version, (128, 256, 64), hip_stream)


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


@pytest.mark.parametrize("shape", [(128, 128, 16), (128, 256, 32), (256, 128, 48),
                                  (256, 384, 128), (128, 128, 1024)])
def test_opt_gemm(hip, opt_gemm, shape):
    dtype, launch = opt_gemm
    _check_gemm(hip, launch, None, shape, dtype=dtype)


@pytest.mark.parametrize("tile_count", [1, 3])
def test_opt_gemm_short_pipeline(hip, opt_gemm, tile_count):
    dtype, launch = opt_gemm
    tile_k = 8 if dtype == np.float32 else 16
    _check_gemm(hip, launch, None, (128, 128, tile_count * tile_k), dtype=dtype)


def test_opt_gemm_nondefault_stream(hip, opt_gemm, hip_stream):
    dtype, launch = opt_gemm
    _check_gemm(hip, launch, None, (128, 256, 64), hip_stream, dtype=dtype)


def test_opt_gemm_fp32_accumulation(hip, opt_gemm):
    # FP16 逐次累加 1 在 2048 后停止增长, FP32 累加应得到 4096.
    dtype, launch = opt_gemm
    M, N, K = 128, 128, 4096
    A = np.ones((M, K), dtype=dtype)
    B = np.ones((K, N), dtype=dtype)
    with (hip.from_numpy(A) as device_A, hip.from_numpy(B) as device_B,
          GuardedOutput(hip, (M, N), dtype=dtype) as output):
        hip.check(launch(device_A.ptr, device_B.ptr, output.ptr, M, N, K, None))
        np.testing.assert_array_equal(output.read(), K)


def test_opt_gemm_empty(opt_gemm):
    _, launch = opt_gemm
    assert launch(None, None, None, 0, 128, 16, None) == 0


@pytest.mark.parametrize("shape", [(127, 128, 16), (128, 129, 16), (128, 128, 17)])
def test_opt_gemm_partial_tile_rejected(hip, opt_gemm, shape):
    _, launch = opt_gemm
    with hip.buffer(16) as data:
        assert launch(data.ptr, data.ptr, data.ptr, *shape, None) != 0


def test_opt_gemm_misaligned_rejected(hip, opt_gemm):
    dtype, launch = opt_gemm
    with hip.buffer(16) as data:
        unaligned = ctypes.c_void_p(data.ptr.value + np.dtype(dtype).itemsize)
        assert launch(unaligned, data.ptr, data.ptr, 128, 128, 16, None) != 0


HGEMM_VERSIONS = list(range(1, 9))


@pytest.fixture(scope="module")
def hgemm():
    return bind_launch("matmul/hgemm", "launch_hgemm", [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ])


def _check_hgemm(hip, launch, version, shape, stream=None, offset=0, fractional=False):
    M, N, K = shape
    rng = np.random.default_rng(2026)
    # 小整数点积用小容差核对, 检查 lane 映射, 列主序布局和流水线.
    A = rng.choice([-1, 1], (M, K)).astype(np.float16)
    B = rng.choice([-1, 1], (K, N)).astype(np.float16)
    if fractional:
        A = rng.uniform(-0.125, 0.125, (M, K)).astype(np.float16)
        B = rng.uniform(-0.125, 0.125, (K, N)).astype(np.float16)
    padded_A = np.concatenate([np.full(offset, 85, dtype=np.float16), A.ravel()])
    # B 与 C 的物理缓冲区按列主序展开, CPU 参考仍使用逻辑 [K,N] 与 [M,N].
    padded_B = np.concatenate([np.full(offset, 85, dtype=np.float16), B.ravel(order="F")])
    with (hip.from_numpy(padded_A) as device_A, hip.from_numpy(padded_B) as device_B,
          GuardedOutput(hip, (M * N,), dtype=np.float16) as output):
        ptr_A = ctypes.c_void_p(device_A.ptr.value + offset * 2)
        ptr_B = ctypes.c_void_p(device_B.ptr.value + offset * 2)
        hip.check(launch(ptr_A, ptr_B, output.ptr, M, N, K, version, stream))
        actual = output.read().reshape((M, N), order="F").copy()
        if fractional:
            # 优化版本须保持与单 wave V1 相同的 WMMA 累加次序和结果.
            hip.check(launch(ptr_A, ptr_B, output.ptr, M, N, K, 1, stream))
            np.testing.assert_array_equal(actual, output.read().reshape((M, N), order="F"))
        after_A, after_B = np.empty_like(padded_A), np.empty_like(padded_B)
        device_A.copy_to(after_A)
        device_B.copy_to(after_B)
        np.testing.assert_array_equal(after_A, padded_A)
        np.testing.assert_array_equal(after_B, padded_B)
    expected = A.astype(np.float64) @ B.astype(np.float64)
    if fractional:
        # FP16 WMMA 的内部舍入与 FP64 点积不同, 长 K 用 FP16 精度容差.
        np.testing.assert_allclose(actual, expected, rtol=0.02, atol=0.005)
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.01
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("version", HGEMM_VERSIONS)
@pytest.mark.parametrize("shape", [(256, 256, 16), (256, 512, 48), (256, 256, 192)])
def test_hgemm(hip, hgemm, version, shape):
    _check_hgemm(hip, hgemm, version, shape)


@pytest.mark.parametrize("version", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 23, 7), (127, 95, 63),
                                  (129, 97, 65), (257, 513, 129), (1153, 193, 65)])
def test_hgemm_partial_tiles(hip, hgemm, version, shape):
    _check_hgemm(hip, hgemm, version, shape)


@pytest.mark.parametrize("version", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("shape", [(129, 97, 65), (256, 384, 128)])
def test_hgemm_offset_inputs(hip, hgemm, version, shape):
    _check_hgemm(hip, hgemm, version, shape, offset=1)


@pytest.mark.parametrize("version", HGEMM_VERSIONS)
def test_hgemm_nondefault_stream(hip, hgemm, hip_stream, version):
    _check_hgemm(hip, hgemm, version, (256, 256, 128), stream=hip_stream)


@pytest.mark.parametrize("version", HGEMM_VERSIONS)
def test_hgemm_fractional_inputs(hip, hgemm, version):
    _check_hgemm(hip, hgemm, version, (256, 256, 1024), fractional=True)


@pytest.mark.parametrize("version", HGEMM_VERSIONS)
def test_hgemm_empty(hgemm, version):
    assert hgemm(None, None, None, 0, 256, 16, version, None) == 0


@pytest.mark.parametrize("version", [1, 2, 3])
def test_hgemm_partial_tile_rejected(hip, hgemm, version):
    with hip.buffer(16) as data:
        assert hgemm(data.ptr, data.ptr, data.ptr, 17, 23, 7, version, None) != 0


@pytest.mark.parametrize("version", [2, 3])
def test_hgemm_misaligned_rejected(hip, hgemm, version):
    with hip.buffer(16) as data:
        unaligned = ctypes.c_void_p(data.ptr.value + 2)
        assert hgemm(unaligned, data.ptr, data.ptr, 256, 256, 16, version, None) != 0


@pytest.mark.parametrize("shape", [(-1, 256, 16), (256, -1, 16), (256, 256, 0),
                                  (65536, 65536, 65536)])
def test_hgemm_invalid_shape(hgemm, shape):
    assert hgemm(None, None, None, *shape, 8, None) != 0


@pytest.mark.parametrize("version", [0, 9])
def test_hgemm_unknown_version(hgemm, version):
    assert hgemm(None, None, None, 256, 256, 16, version, None) != 0


@pytest.fixture(scope="module")
def hgemm_variants(tmp_path_factory):
    """额外实例化非恒等 XOR 和 V8 双缓冲, 避免只验证默认模板参数."""
    import os
    import subprocess

    from common.python.sdk import REPO_ROOT, sdk_root

    directory = tmp_path_factory.mktemp("hgemm_variants")
    source = directory / "variants.hip"
    source.write_text(f'#include "{REPO_ROOT}/hip/matmul/hgemm.hip"\n' + r'''
extern "C" hipError_t launch_variant(const __half* A, const __half* B, __half* C,
                                     int M, int N, int K, int version, hipStream_t stream) {
  if (version == 0) {
    hgemm_v3_kernel<128, 128, 64, 64, 64, 0, true>
      <<<dim3(N / 128, M / 128), 128, 0, stream>>>(A, B, C, M, N, K);
  } else {
    hgemm_v8_kernel<128, 96, 64, 64, 48, 8, 8, 2>
      <<<dim3((N + 95) / 96, (M + 127) / 128), 128, 0, stream>>>(A, B, C, M, N, K);
  }
  return hipGetLastError();
}
''')
    library = directory / "variants.so"
    mode = "-mcumode" if os.environ.get("GPU_MODE") == "cu" else "-mno-cumode"
    subprocess.run([
        str(REPO_ROOT / ".venv/bin/hipcc"), "-O3", "-std=c++17", mode,
        f'--offload-arch={os.environ.get("GPU_ARCH", "gfx1151")}', "-shared", "-fPIC",
        str(source), "-o", str(library),
    ], env={**os.environ, "ROCM_PATH": str(sdk_root())}, check=True, capture_output=True)
    loaded = ctypes.CDLL(str(library))
    launch = loaded.launch_variant
    launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    launch.restype = ctypes.c_int
    return launch


@pytest.mark.parametrize("shape", [(128, 128, 64), (256, 384, 192)])
def test_hgemm_xor_layout(hip, hgemm_variants, shape):
    _check_hgemm(hip, hgemm_variants, 0, shape)


@pytest.mark.parametrize("shape", [(128, 96, 64), (257, 193, 193), (1153, 193, 65)])
def test_hgemm_v8_double_buffer(hip, hgemm_variants, shape):
    _check_hgemm(hip, hgemm_variants, 1, shape)
