import numpy as np
import ctypes

from .ojph_bindings import Codestream, J2COutfile, MemOutfile, Point


class CompressedData:
    def __init__(self, mem_file, codestream):
        self._mem_file = mem_file
        self._codestream = codestream

    def __del__(self):
        """Clean up the memory file when this object is garbage collected."""
        if self._codestream is not None:
            self._codestream.close()
        self._codestream = None
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
        return self._mem_file.get_used_size()

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

def imwrite_to_memory(image, *, channel_order=None):
    mem_outfile = MemOutfile()
    mem_outfile.open(65536, False)
    codestream = imwrite(mem_outfile, image, channel_order=channel_order)
    return CompressedData(mem_outfile, codestream)


def imwrite(filename, image, *, channel_order=None):
    # Auto-detect channel order if not provided
    if channel_order is None:
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

    # Validate channel order format
    valid_orders = {'HW', 'HWC', 'CHW'}
    if channel_order not in valid_orders:
        raise ValueError(
            f"Invalid channel_order '{channel_order}'. "
            f"Must be one of: {', '.join(valid_orders)}"
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

    siz.set_image_extent(Point(width, height))
    if 'C' in channel_order:
        num_components = image.shape[channel_order.index('C')]
    else:
        num_components = 1

    bit_depth = image.dtype.itemsize * 8
    is_signed = image.dtype.kind != 'u'
    siz.set_num_components(num_components)
    for i in range(num_components):
        siz.set_component(
            i,
            Point(1, 1), # component downsampling
            bit_depth,
            is_signed,
        )
    cod = codestream.access_cod()
    cod.set_reversible(True)

    # Enable color transform for RGB/RGBA images in HWC format (3 or 4 components)
    # This enables automatic RGB detection via is_employing_color_transform()
    if num_components in [3, 4] and channel_order == 'HWC':
        cod.set_color_transform(True)
        # Color transform requires non-planar mode
        codestream.set_planar(False)
    else:
        # Use planar mode for better efficiency with other multi-component images
        # This processes each component separately, improving cache locality
        codestream.set_planar(num_components > 1)

    codestream.write_headers(ojph_file, None, 0)
    c = 0
    line = codestream.exchange(None, c)

    if num_components == 1:
        # Single component - simple case
        for i in range(height):
            i32_ptr = ctypes.cast(line.i32_address, ctypes.POINTER(ctypes.c_uint32))
            line_array = np.ctypeslib.as_array(
                ctypes.cast(i32_ptr, ctypes.POINTER(ctypes.c_uint32)),
                shape=(line.size,)
            )
            line_array[...] = image[i, :]
            line = codestream.exchange(line, 0)
    else:
        # Multi-component - use planar mode for efficiency
        if channel_order == 'HWC':
            # HWC format: image[height, width, channel]
            for c in range(num_components):
                for i in range(height):
                    i32_ptr = ctypes.cast(line.i32_address, ctypes.POINTER(ctypes.c_uint32))
                    line_array = np.ctypeslib.as_array(
                        ctypes.cast(i32_ptr, ctypes.POINTER(ctypes.c_uint32)),
                        shape=(line.size,)
                    )
                    line_array[...] = image[i, :, c]
                    line = codestream.exchange(line, c)
        elif channel_order == 'CHW':
            # CHW format: image[channel, height, width]
            for c in range(num_components):
                for i in range(height):
                    i32_ptr = ctypes.cast(line.i32_address, ctypes.POINTER(ctypes.c_uint32))
                    line_array = np.ctypeslib.as_array(
                        ctypes.cast(i32_ptr, ctypes.POINTER(ctypes.c_uint32)),
                        shape=(line.size,)
                    )
                    line_array[...] = image[c, i, :]
                    line = codestream.exchange(line, c)

    codestream.flush()
    if not is_mem_file:
        codestream.close()
    else:
        return codestream
