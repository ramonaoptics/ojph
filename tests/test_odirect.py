import os
import sys
import pytest
import numpy as np

from ojph.ojph_bindings import J2CInfileWithFlags, Codestream
from ojph._imwrite import imwrite_to_memory


@pytest.mark.skipif(sys.platform != "linux", reason="O_DIRECT is Linux-specific")
def test_j2c_infile_with_flags_basic(tmp_path):
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    filename = tmp_path / 'test.j2c'
    compressed_data = imwrite_to_memory(test_image)
    with open(filename, 'wb') as f:
        f.write(compressed_data)

    infile = J2CInfileWithFlags()
    infile.open(str(filename), 0)

    codestream = Codestream()
    codestream.read_headers(infile)

    siz = codestream.access_siz()
    extents = siz.get_image_extent()
    assert extents.x == 64
    assert extents.y == 64

    infile.close()


@pytest.mark.skipif(sys.platform != "linux", reason="O_DIRECT is Linux-specific")
def test_j2c_infile_with_odirect(tmp_path):
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)

    filename = tmp_path / 'test.j2c'
    compressed_data = imwrite_to_memory(test_image)
    with open(filename, 'wb') as f:
        f.write(compressed_data)

    O_DIRECT = os.O_DIRECT
    infile = J2CInfileWithFlags()
    infile.open(str(filename), O_DIRECT)

    codestream = Codestream()
    codestream.read_headers(infile)

    siz = codestream.access_siz()
    extents = siz.get_image_extent()
    assert extents.x == 64
    assert extents.y == 64

    infile.close()
