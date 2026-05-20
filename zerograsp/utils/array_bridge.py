import numpy as np
import torch as th


_NUMPY_TO_TORCH_DTYPE = {
    np.dtype(np.float32): th.float32,
    np.dtype(np.float64): th.float64,
    np.dtype(np.int32): th.int32,
    np.dtype(np.int64): th.int64,
    np.dtype(np.uint8): th.uint8,
    np.dtype(np.bool_): th.bool,
}

_TORCH_TO_NUMPY_DTYPE = {
    th.float32: np.float32,
    th.float64: np.float64,
    th.int32: np.int32,
    th.int64: np.int64,
    th.uint8: np.uint8,
    th.bool: np.bool_,
}


def numpy_to_torch(array, dtype=None, device=None):
    arr = np.asarray(array)
    torch_dtype = dtype or _NUMPY_TO_TORCH_DTYPE.get(arr.dtype)
    if torch_dtype is None:
        raise TypeError(f"Unsupported numpy dtype: {arr.dtype}")

    target_numpy_dtype = _TORCH_TO_NUMPY_DTYPE[torch_dtype]
    arr = np.ascontiguousarray(arr.astype(target_numpy_dtype, copy=False))
    buffer = bytearray(arr.tobytes())
    tensor = th.frombuffer(buffer, dtype=torch_dtype).clone().reshape(arr.shape)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def torch_to_numpy(tensor):
    t = tensor.detach().cpu().contiguous()
    numpy_dtype = _TORCH_TO_NUMPY_DTYPE.get(t.dtype)
    if numpy_dtype is None:
        raise TypeError(f"Unsupported torch dtype: {t.dtype}")
    return np.asarray(t.tolist(), dtype=numpy_dtype)
