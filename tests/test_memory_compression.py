import numpy as np
import tempfile
import os
from ojph._imwrite import imwrite, imwrite_to_memory


def test_imwrite_to_memory_grayscale():
    test_image = np.random.randint(0, 256, (100, 150), dtype=np.uint8)

    compressed_data = imwrite_to_memory(test_image)

    # Check that it's a CompressedData object
    array = np.asarray(compressed_data)
    assert not array.flags.writeable
    assert len(compressed_data) > 0

    # Convert to bytes for file writing
    compressed_bytes = compressed_data.tobytes()

    with tempfile.NamedTemporaryFile(suffix='.j2c', delete=False) as temp_file:
        temp_file.write(compressed_bytes)
        temp_filename = temp_file.name

    try:
        from ojph._imread import imread
        loaded_image = imread(temp_filename)
        np.testing.assert_array_equal(loaded_image, test_image)
    finally:
        os.unlink(temp_filename)


def test_imwrite_to_memory_color():
    test_image = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)

    compressed_data = imwrite_to_memory(test_image)

    # Check that it's a CompressedData object
    array = np.asarray(compressed_data)
    assert not array.flags.writeable
    assert len(compressed_data) > 0

    # Convert to bytes for file writing
    compressed_bytes = compressed_data.tobytes()

    with tempfile.NamedTemporaryFile(suffix='.j2c', delete=False) as temp_file:
        temp_file.write(compressed_bytes)
        temp_filename = temp_file.name

    try:
        from ojph._imread import imread
        loaded_image = imread(temp_filename)
        np.testing.assert_array_equal(loaded_image, test_image)
    finally:
        os.unlink(temp_filename)


def test_imwrite_to_memory_16bit():
    test_image = np.random.randint(0, 65536, (50, 75), dtype=np.uint16)

    compressed_data = imwrite_to_memory(test_image)

    # Check that it's a CompressedData object
    array = np.asarray(compressed_data)
    assert not array.flags.writeable
    assert len(compressed_data) > 0

    # Convert to bytes for file writing
    compressed_bytes = compressed_data.tobytes()

    with tempfile.NamedTemporaryFile(suffix='.j2c', delete=False) as temp_file:
        temp_file.write(compressed_bytes)
        temp_filename = temp_file.name

    try:
        from ojph._imread import imread
        loaded_image = imread(temp_filename)
        np.testing.assert_array_equal(loaded_image, test_image)
    finally:
        os.unlink(temp_filename)


def test_imwrite_to_memory_consistency():
    test_image = np.random.randint(0, 256, (80, 120), dtype=np.uint8)

    compressed_data_memory = imwrite_to_memory(test_image)

    with tempfile.NamedTemporaryFile(suffix='.j2c', delete=False) as temp_file:
        imwrite(temp_file.name, test_image)
        with open(temp_file.name, 'rb') as f:
            compressed_data_file = f.read()

    try:
        # Convert memory data to bytes for comparison
        compressed_bytes_memory = compressed_data_memory.tobytes()
        assert compressed_bytes_memory == compressed_data_file
    finally:
        os.unlink(temp_file.name)
