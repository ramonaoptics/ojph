import os
import sys
import pytest
import numpy as np

from ojph import imread, imwrite


@pytest.mark.skipif(sys.platform != "linux", reason="O_DIRECT is Linux-specific")
def test_imread_with_flags(tmp_path):
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    filename = tmp_path / 'test.j2c'
    imwrite(filename, test_image)

    O_DIRECT = os.O_DIRECT
    decoded_image = imread(filename, flags=O_DIRECT)

    assert np.array_equal(test_image, decoded_image)
    assert decoded_image.shape == test_image.shape
    assert decoded_image.dtype == test_image.dtype


@pytest.mark.skipif(sys.platform != "linux", reason="O_DIRECT is Linux-specific")
def test_imwrite_with_flags(tmp_path):
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    filename = tmp_path / 'test.j2c'
    O_DIRECT = os.O_DIRECT
    imwrite(filename, test_image, flags=O_DIRECT)

    decoded_image = imread(filename)

    assert np.array_equal(test_image, decoded_image)
    assert decoded_image.shape == test_image.shape
    assert decoded_image.dtype == test_image.dtype


def test_imread_imwrite_without_flags(tmp_path):
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    filename = tmp_path / 'test.j2c'
    imwrite(filename, test_image)

    decoded_image = imread(filename)

    assert np.array_equal(test_image, decoded_image)
    assert decoded_image.shape == test_image.shape
    assert decoded_image.dtype == test_image.dtype
