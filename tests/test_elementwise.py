"""Correctness tests for elementwise operators."""

import ctypes

import ml_dtypes
import numpy as np
import pytest

from tests.helpers import GuardedOutput, bind_launch


ELEMENT_COUNTS = [
    0, 1, 7, 8, 9, 31, 32, 33, 255, 256, 257, 1025,
    2047, 2048, 2049, 1 << 20, 1 << 24,
]
BF16_ATOL = float(ml_dtypes.finfo(ml_dtypes.bfloat16).smallest_subnormal)


def _silu_reference(x):
    """使用 FP64 稳定 sigmoid 计算数学参考, 避免依赖设备端 expf 实现."""
    values = x.astype(np.float64)
    exponential = np.exp(-np.abs(values))
    sigmoid = np.where(values >= 0, 1 / (1 + exponential),
                       exponential / (1 + exponential))
    return values * sigmoid


def _silu_mul_reference(gate, up):
    act = _silu_reference(gate).astype(ml_dtypes.bfloat16).astype(np.float32)
    return act * up


@pytest.fixture(scope="module", params=["add", "mul", "silu_mul"])
def binary(request):
    name = request.param
    launch = bind_launch(f"elementwise/{name}", f"launch_{name}", [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_void_p,
    ])
    reference = {
        "add": np.add, "mul": np.multiply, "silu_mul": _silu_mul_reference,
    }[name]
    return launch, reference


def _check_binary(hip, binary, a, b, a_offset=0, b_offset=0, y_offset=0,
                  stream=None):
    launch, reference = binary
    a_storage = np.full(a.size + a_offset, 85, dtype=a.dtype)
    b_storage = np.full(b.size + b_offset, 85, dtype=b.dtype)
    a_storage[a_offset:] = a.ravel()
    b_storage[b_offset:] = b.ravel()
    with (
        hip.from_numpy(a_storage) as device_a,
        hip.from_numpy(b_storage) as device_b,
        GuardedOutput(hip, (a.size + y_offset,), a.dtype) as output,
    ):
        a_ptr = ctypes.c_void_p(device_a.ptr.value + a_offset * a.itemsize)
        b_ptr = ctypes.c_void_p(device_b.ptr.value + b_offset * b.itemsize)
        y_ptr = ctypes.c_void_p(output.ptr.value + y_offset * a.itemsize)
        hip.check(launch(a_ptr, b_ptr, y_ptr, a.size, stream))
        storage = output.read()
        np.testing.assert_array_equal(storage[:y_offset], output.sentinel)
        actual = storage[y_offset:].reshape(a.shape)

    with np.errstate(over="ignore", invalid="ignore"):
        expected = reference(a.astype(np.float32), b.astype(np.float32)).astype(
            ml_dtypes.bfloat16,
        )
    # BF16 值可精确转换为 FP32, 便于 NumPy 正确比较 NaN 等特殊值.
    if reference is _silu_mul_reference:
        np.testing.assert_allclose(
            actual.astype(np.float32), expected.astype(np.float32),
            rtol=1 / 128, atol=BF16_ATOL,
        )
    else:
        np.testing.assert_array_equal(
            actual.astype(np.float32), expected.astype(np.float32),
        )


@pytest.mark.parametrize("n", ELEMENT_COUNTS)
def test_binary(hip, binary, n):
    rng = np.random.default_rng(42)
    a = rng.standard_normal(n, dtype=np.float32).astype(ml_dtypes.bfloat16)
    b = rng.standard_normal(n, dtype=np.float32).astype(ml_dtypes.bfloat16)
    _check_binary(hip, binary, a, b)


@pytest.mark.parametrize("n", [7, 8, 9, 2047, 2048, 2049])
@pytest.mark.parametrize("offsets", [(1, 0, 0), (0, 2, 0), (0, 0, 4)])
def test_binary_offset_buffers(hip, binary, n, offsets):
    # 输入和输出分别偏移, 覆盖较窄向量路径及向量尾部的边界.
    rng = np.random.default_rng(2026)
    a = rng.standard_normal(n, dtype=np.float32).astype(ml_dtypes.bfloat16)
    b = rng.standard_normal(n, dtype=np.float32).astype(ml_dtypes.bfloat16)
    _check_binary(hip, binary, a, b, *offsets)


def test_binary_null_pointer(hip, binary):
    # 空输入允许空指针, 非空输入应在启动前拒绝空指针.
    launch, _ = binary
    hip.check(launch(None, None, None, 0, None))
    assert launch(None, None, None, 1, None) != 0
    with hip.buffer(16) as data:
        for pointers in [(None, data.ptr, data.ptr),
                         (data.ptr, None, data.ptr),
                         (data.ptr, data.ptr, None)]:
            assert launch(*pointers, 1, None) != 0


def test_binary_special_values(hip, binary):
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
    _check_binary(hip, binary, a, b)


def test_binary_grid_overflow(hip, binary):
    launch, _ = binary
    with hip.buffer(16) as data:
        assert launch(data.ptr, data.ptr, data.ptr,
                      ctypes.c_size_t(-1).value, None) != 0


def test_binary_nondefault_stream(hip, binary, hip_stream):
    rng = np.random.default_rng(2026)
    a = rng.standard_normal(2049, dtype=np.float32).astype(ml_dtypes.bfloat16)
    b = rng.standard_normal(2049, dtype=np.float32).astype(ml_dtypes.bfloat16)
    _check_binary(hip, binary, a, b, stream=hip_stream)


@pytest.fixture(scope="module")
def silu():
    return bind_launch("elementwise/silu", "launch_silu", [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
    ])


def _check_silu(hip, silu, x, x_offset=0, y_offset=0, stream=None):
    host = np.full(x.size + x_offset, 85, dtype=x.dtype)
    host[x_offset:] = x.ravel()
    with (
        hip.from_numpy(host) as device_x,
        GuardedOutput(hip, (x.size + y_offset,), x.dtype) as output,
    ):
        x_ptr = ctypes.c_void_p(device_x.ptr.value + x_offset * x.itemsize)
        y_ptr = ctypes.c_void_p(output.ptr.value + y_offset * x.itemsize)
        hip.check(silu(x_ptr, y_ptr, x.size, stream))
        storage = output.read()
        np.testing.assert_array_equal(storage[:y_offset], output.sentinel)
        actual = storage[y_offset:].reshape(x.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = _silu_reference(x).astype(ml_dtypes.bfloat16)
    # expf 的精度与 FP64 参考不同, 按 BF16 的相对精度设置误差容限.
    np.testing.assert_allclose(
        actual.astype(np.float32), expected.astype(np.float32),
        rtol=1 / 128, atol=BF16_ATOL,
    )


@pytest.mark.parametrize("n", ELEMENT_COUNTS)
def test_silu(hip, silu, n):
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n, dtype=np.float32).astype(ml_dtypes.bfloat16)
    _check_silu(hip, silu, x)


@pytest.mark.parametrize("n", [7, 8, 9, 2047, 2048, 2049])
@pytest.mark.parametrize("offsets", [(1, 0), (0, 2), (0, 4)])
def test_silu_offset_buffers(hip, silu, n, offsets):
    rng = np.random.default_rng(2026)
    x = rng.standard_normal(n, dtype=np.float32).astype(ml_dtypes.bfloat16)
    _check_silu(hip, silu, x, *offsets)


def test_silu_null_pointer(hip, silu):
    hip.check(silu(None, None, 0, None))
    assert silu(None, None, 1, None) != 0
    with hip.buffer(16) as data:
        assert silu(None, data.ptr, 1, None) != 0
        assert silu(data.ptr, None, 1, None) != 0


def test_silu_special_values(hip, silu):
    limit = ml_dtypes.finfo(ml_dtypes.bfloat16)
    x = np.array([
        -1000, -80, -20, -1, -0.0, 0.0, 1, 20, 80, 1000,
        -limit.max, limit.max, -limit.tiny, limit.tiny,
        -np.inf, np.inf, np.nan,
    ], dtype=ml_dtypes.bfloat16)
    _check_silu(hip, silu, x)


def test_silu_grid_overflow(hip, silu):
    with hip.buffer(16) as data:
        assert silu(data.ptr, data.ptr, ctypes.c_size_t(-1).value, None) != 0


def test_silu_nondefault_stream(hip, silu, hip_stream):
    rng = np.random.default_rng(2026)
    x = rng.standard_normal(2049, dtype=np.float32).astype(ml_dtypes.bfloat16)
    _check_silu(hip, silu, x, stream=hip_stream)


@pytest.mark.parametrize("n", [1, 8, 9, 257])
def test_silu_mul_intermediate_rounding(hip, n):
    # 此输入可区分先舍入 SiLU 和直接以 FP32 激活乘以 up 的两条路径.
    launch = bind_launch("elementwise/silu_mul", "launch_silu_mul", [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_void_p,
    ])
    values = np.full(n, 1.25, dtype=ml_dtypes.bfloat16)
    expected = _silu_mul_reference(values.astype(np.float32),
                                   values.astype(np.float32)).astype(ml_dtypes.bfloat16)
    unrounded = (_silu_reference(values) * values.astype(np.float64)).astype(
        ml_dtypes.bfloat16,
    )
    assert np.all(expected != unrounded)
    with (
        hip.from_numpy(values) as gate,
        hip.from_numpy(values) as up,
        GuardedOutput(hip, (n,), values.dtype) as output,
    ):
        hip.check(launch(gate.ptr, up.ptr, output.ptr, n, None))
        np.testing.assert_array_equal(output.read().astype(np.float32),
                                      expected.astype(np.float32))


@pytest.fixture(scope="module")
def softmax():
    return bind_launch("elementwise/softmax", "launch_softmax", [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ])


def _check_softmax(hip, softmax, x, x_offset=0, y_offset=0, stream=None):
    x = np.ascontiguousarray(x, dtype=ml_dtypes.bfloat16)
    rows, cols = x.shape
    storage = np.full(x.size + x_offset + 16, 85, dtype=x.dtype)
    storage[x_offset:x_offset + x.size] = x.ravel()
    with (
        hip.from_numpy(storage) as device_x,
        GuardedOutput(hip, (x.size + y_offset,), x.dtype) as output,
    ):
        x_ptr = ctypes.c_void_p(device_x.ptr.value + x_offset * x.itemsize)
        y_ptr = ctypes.c_void_p(output.ptr.value + y_offset * x.itemsize)
        hip.check(softmax(x_ptr, y_ptr, rows, cols, stream))
        result = output.read()
        np.testing.assert_array_equal(result[:y_offset], output.sentinel)
        actual = result[y_offset:].reshape(x.shape).astype(np.float32)
        after = np.empty_like(storage)
        device_x.copy_to(after)
        np.testing.assert_array_equal(after, storage)

    # 以 FP64 计算数学参考, 再舍入到 BF16, 同时检查概率和掩码语义.
    values = x.astype(np.float64)
    exponentials = np.exp(values - values.max(axis=1, keepdims=True))
    expected = (exponentials / exponentials.sum(axis=1, keepdims=True)).astype(
        ml_dtypes.bfloat16,
    ).astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=1 / 128, atol=BF16_ATOL)
    assert np.isfinite(actual).all()
    assert ((actual >= 0) & (actual <= 1)).all()
    np.testing.assert_array_equal(actual[np.isneginf(values)], 0)
    np.testing.assert_allclose(
        actual.astype(np.float64).sum(axis=1), 1, rtol=0, atol=0.0045,
    )


@pytest.mark.parametrize("cols", [
    1, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    255, 256, 257, 5120, 16385, 131073,
])
@pytest.mark.parametrize("rows", [1, 5])
@pytest.mark.parametrize("offsets", [(0, 0), (1, 3)])
def test_softmax(hip, softmax, cols, rows, offsets):
    rng = np.random.default_rng(2026)
    _check_softmax(hip, softmax, rng.standard_normal((rows, cols)), *offsets)


@pytest.mark.parametrize("case", [
    "uniform", "negative_uniform", "partial_mask", "single_valid",
    "large_scores", "shifted_scores",
])
def test_softmax_special_rows(hip, softmax, case):
    rng = np.random.default_rng(2026)
    if case == "uniform":
        x = np.zeros((3, 257))
    elif case == "negative_uniform":
        x = np.full((3, 33), -500.0)
    elif case == "partial_mask":
        x = rng.standard_normal((3, 257))
        x[:, ::3] = -np.inf
    elif case == "single_valid":
        x = np.full((3, 257), -np.inf)
        x[np.arange(3), [0, 128, 256]] = [0, 5, -5]
    elif case == "large_scores":
        x = np.array([
            [1000, 990, -1000, -np.inf],
            [-1000, -990, -1010, -np.inf],
        ])
    else:
        x = rng.standard_normal((3, 65)) * 3 + 64
    _check_softmax(hip, softmax, x, 1, 3)


@pytest.mark.parametrize("cols", [4, 33, 65, 129])
def test_softmax_nondefault_stream(hip, softmax, hip_stream, cols):
    rng = np.random.default_rng(2026)
    _check_softmax(hip, softmax, rng.standard_normal((5, cols)), 1, 3,
                   stream=hip_stream)


def test_softmax_empty(hip, softmax):
    hip.check(softmax(None, None, 0, 1, None))


@pytest.mark.parametrize("shape", [(-1, 4), (1, 0), (1, -1), (0, 0), (0, -1)])
def test_softmax_invalid_shape(softmax, shape):
    assert softmax(None, None, *shape, None) != 0


@pytest.mark.parametrize("missing", ["x", "y", "both"])
def test_softmax_null_pointer(hip, softmax, missing):
    with hip.buffer(16) as data:
        x_ptr = None if missing in ("x", "both") else data.ptr
        y_ptr = None if missing in ("y", "both") else data.ptr
        assert softmax(x_ptr, y_ptr, 1, 4, None) != 0
