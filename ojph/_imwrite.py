import numpy as np
import ctypes

from .ojph_bindings import Codestream, J2COutfile, MemOutfile, Point


class CompressedData:
    def __init__(self, mem_file):
        self._mem_file = mem_file

    def __del__(self):
        """Clean up the memory file when this object is garbage collected."""
        if self._mem_file is not None:
            self._mem_file.close()
        self._mem_file = None

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        return np.asarray(self, copy=True)[key]

    def copy(self):
        """Create a copy of the data."""
        return np.asarray(self, copy=True)

    def tobytes(self):
        """Convert to bytes."""
        return np.asarray(self).tobytes()

    @property
    def size(self):
        return self._mem_file.tell()

    @property
    def shape(self):
        return (self.size,)

    @property
    def dtype(self):
        return np.uint8

    def __array__(self, dtype=None, *, copy=None):
        """Convert to numpy array."""
        if copy is not None and not copy:
            raise ValueError(f"A copy is required for operations on {self.__class__}")

        memoryview = self._mem_file.get_data()
        assert memoryview.readonly, "Memoryview must be readonly"
        array = np.asarray(memoryview, dtype=np.uint8)
        if dtype is not None:
            array = array.astype(dtype)
        return array

def imwrite_to_memory(image):
    mem_outfile = MemOutfile()
    mem_outfile.open(65536, False)
    imwrite(mem_outfile, image)
    return CompressedData(mem_outfile)


def imwrite(filename, image):
    # In the future we might be able to pass in a channel_order parameter
    if image.ndim == 2:
        channel_order = 'HW'
    else:
        channel_order = 'HWC'

    channel_order = channel_order.upper()

    if len(channel_order) != image.ndim:
        raise ValueError(
            f"The channel order ({channel_order}) must be consistent "
            f"with the image dimensions ({image.ndim})."
        )

    if isinstance(filename, MemOutfile):
        ojph_file = filename
        is_mem_file = True
    else:
        ojph_file = J2COutfile()
        ojph_file.open(str(filename))
        is_mem_file = False
    codestream = Codestream()

    siz = codestream.access_siz()
    width = image.shape[channel_order.index('W')]
    height = image.shape[channel_order.index('H')]

    # What does planar mean? it doesn't seem to mean components...
    # is_planar = 'C' in channel_order
    siz.set_image_extent(Point(width, height))
    if 'C' in channel_order:
        num_components = image.shape[channel_order.index('C')]
    else:
        num_components = 1

    bit_depth = image.dtype.itemsize * 8
    # Is there a better way to detect signed dtypes???
    is_signed = image.dtype.kind != 'u'
    siz.set_num_components(num_components)
    for i in range(num_components):
        # is it necessary to do this in a loop?
        siz.set_component(
            i,
            Point(1, 1), # component downsampling
            bit_depth,
            is_signed,
        )
    cod = codestream.access_cod()
    # cod.set_progression_oder
    # code.set_color_Transform
    cod.set_reversible(True)
    # codestream.set_profile()

    # set tile_size
    # set tile offset
    # planar true is likely for things like
    # YUV420 where the Y, U, and V planes are stored
    # sparately
    codestream.set_planar(False)

    codestream.write_headers(ojph_file, None, 0)
    c = 0
    line = codestream.exchange(None, c)
    for i in range(height):
        for c in range(num_components):
            i32_ptr = ctypes.cast(line.i32_address, ctypes.POINTER(ctypes.c_uint32))
            # Calculate the total number of bytes (size of the array in elements * size of int32)
            line_array = np.ctypeslib.as_array(
                ctypes.cast(i32_ptr, ctypes.POINTER(ctypes.c_uint32)),
                shape=(line.size,)
            )
            if image.ndim == 2:
                line_array[...] = image[i, :]
            else:
                line_array[...] = image[i, :, c]
            # c_before = c
            line = codestream.exchange(line, c)

    codestream.flush()
    if not is_mem_file:
        # This will close the file as well
        # We do not want to close the memfile
        codestream.close()
