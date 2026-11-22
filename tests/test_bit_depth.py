import numpy as np
import pytest

from ojph import imread, imread_from_memory, imwrite, imwrite_to_memory
from ojph._imread import OJPHImageFile
from ojph.ojph_bindings import read_j2c_into


@pytest.mark.parametrize(
    ('dtype', 'low', 'high', 'bit_depth', 'is_signed'),
    [
        (np.uint8, 0, 1, 2, True),
        (np.uint8, 0, 200, 8, False),
        (np.int8, -100, 27, 8, True),
        (np.uint16, 0, 53, 7, True),
        (np.int16, -3000, 7515, 14, True),
        (np.uint16, 0, 4095, 12, False),
        (np.int32, -70000, 70000, 18, True),
    ],
)
@pytest.mark.parametrize('wavelet', ['rev12', 'rev53'])
def test_declared_bit_depth_round_trips(tmp_path, dtype, low, high, bit_depth, is_signed, wavelet):
    rng = np.random.default_rng(0)
    image = rng.integers(low, high + 1, size=(67, 131), dtype=dtype)
    image[0, 0], image[-1, -1] = low, high
    data = imwrite_to_memory(
        image, wavelet=wavelet, num_decompositions=3,
        bit_depth=bit_depth, is_signed=is_signed,
    )

    decoded = imread_from_memory(data)
    assert decoded.dtype == dtype
    np.testing.assert_array_equal(decoded, image)

    path = tmp_path / 'image.j2c'
    imwrite(path, image, wavelet=wavelet, num_decompositions=3,
            bit_depth=bit_depth, is_signed=is_signed)
    decoded = imread(path)
    assert decoded.dtype == dtype
    np.testing.assert_array_equal(decoded, image)

    comment = f"ojph dtype={np.dtype(dtype).name}".encode()
    raw = bytes(data)
    start = raw.index(comment) - 6
    length = int.from_bytes(raw[start + 2:start + 4], 'big')
    without_comment = np.frombuffer(raw[:start] + raw[start + 2 + length:], dtype=np.uint8)
    reader = OJPHImageFile.from_memory(without_comment)
    assert reader.dtype.kind == ('i' if is_signed else 'u')
    assert reader.dtype.itemsize == (1 if bit_depth <= 8 else 2 if bit_depth <= 16 else 4)
    np.testing.assert_array_equal(
        imread_from_memory(without_comment, out=np.empty_like(image)), image)

    out = np.empty_like(image)
    assert imread_from_memory(data, out=out) is not None
    np.testing.assert_array_equal(out, image)

    out = np.empty_like(image)
    info = np.iinfo(dtype)
    read_j2c_into(
        np.frombuffer(data, dtype=np.uint8), out, 0,
        None if out.itemsize == 4 else int(info.min),
        None if out.itemsize == 4 else int(info.max),
    )
    np.testing.assert_array_equal(out, image)


def test_reading_into_a_narrower_type_saturates():
    image = np.array([[-300, -1, 0, 7, 255, 300]] * 4, dtype=np.int16)
    data = imwrite_to_memory(image, wavelet='rev12', bit_depth=10, is_signed=True)

    out = np.empty(image.shape, dtype=np.uint8)
    imread_from_memory(data, out=out)
    np.testing.assert_array_equal(out, np.clip(image, 0, 255))

    out = np.empty(image.shape, dtype=np.int8)
    imread_from_memory(data, out=out)
    np.testing.assert_array_equal(out, np.clip(image, -128, 127))


@pytest.mark.parametrize(
    ('values', 'bit_depth', 'is_signed'),
    [([0, 2], 2, True), ([-1, 0], 4, False), ([0, 200], None, True)],
)
def test_a_value_outside_the_declared_bit_depth_raises(values, bit_depth, is_signed):
    image = np.array([values] * 2, dtype=np.int16 if min(values) < 0 else np.uint8)
    with pytest.raises(ValueError, match="cannot hold"):
        imwrite_to_memory(image, bit_depth=bit_depth, is_signed=is_signed)
