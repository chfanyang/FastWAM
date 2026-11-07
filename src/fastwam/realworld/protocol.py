"""OpenPI-compatible MessagePack NumPy wire format, independent of robot/GPU imports."""
import msgpack
import numpy as np

VERSION = 1
ACTION_SPACE = "fr3_link8_base_absolute_xyz_wxyz_gripper_open"


def _pack_array(value):
    # Same byte-key format as third_party/FastAPI/package/src/openpi_client/msgpack_numpy.py.
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError("Only numeric arrays are accepted")
        return {b"__ndarray__": True, b"data": value.tobytes(),
                b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _unpack_array(value):
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind not in "biuf":
            raise ValueError("Only numeric arrays are accepted")
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"]).copy()
    return value


def pack(value):
    return msgpack.packb(value, default=_pack_array, use_bin_type=True)


def unpack(value):
    return msgpack.unpackb(value, object_hook=_unpack_array, raw=False)
