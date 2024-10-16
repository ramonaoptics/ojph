from ojph import imwrite, imread
import numpy as np
import pytest


@pytest.mark.parametrize(
    'shape', [
        (200, 100),
        (1024, 256),
        (256, 1024),
        (1024, 1024),
        (1024, 2048),
        (4096, 2048),
        (4096, 4096),
        (8192, 1024),
        (8192, 8192),
    ]
)
def test_write_lossless(shape, tmp_path):
    filename = tmp_path / 'test.jp2'
    data = np.random.randint(0, 256, shape, dtype=np.uint8)
    imwrite(filename, data)
    image_read = imread(filename)

    np.testing.assert_array_equal(data, image_read)
