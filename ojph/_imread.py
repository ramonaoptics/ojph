import numpy as np
import ctypes
from .ojph_bindings import J2CInfile, MemInfile, Codestream
from warnings import warn

# Copy the imageio.v3.imread signature
def imread(uri, *, index=None, plugin=None, extension=None, format_hint=None, **kwargs):
    if index is not None:
        warn(f"index {index} is ignored", stacklevel=2)
    if plugin is not None:
        warn(f"plugin {plugin} is ignored", stacklevel=2)
    if extension is not None:
        warn(f"extension {extension} is ignored", stacklevel=2)
    if format_hint is not None:
        warn(f"format_hint {format_hint} is ignored", stacklevel=2)

    return OJPHImageFile(uri).read_image()


def imread_from_memory(data):
    """
    Read a JPEG2000 image from memory data.

    Parameters
    ----------
    data : numpy.ndarray or bytes-like
        The compressed JPEG2000 data as a numpy array or bytes-like object.

    Returns
    -------
    numpy.ndarray
        The decoded image.
    """
    if isinstance(data, (bytes, bytearray)):
        data = np.frombuffer(data, dtype=np.uint8)
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=np.uint8)

    if data.ndim != 1:
        raise ValueError("Data must be a 1-dimensional array or bytes-like object")

    mem_infile = MemInfile()
    mem_infile.open(data)
    return OJPHImageFile(mem_infile).read_image()


class OJPHImageFile:
    def __init__(self, filename, *, mode='r'):
        if mode != 'r':
            raise ValueError(f"We only support mode = 'r' for now. Got {mode}.")
        self._codestream = None
        self._ojph_file = None
        self._filename = filename
        self._is_mem_file = False
        if isinstance(filename, MemInfile):
            self._ojph_file = filename
            self._is_mem_file = True
        else:
            ojph_file = J2CInfile()
            ojph_file.open(str(filename))
            self._ojph_file = ojph_file

        self._codestream = Codestream()
        self._codestream.read_headers(self._ojph_file)

        siz = self._codestream.access_siz()
        extents = siz.get_image_extent()
        num_components = siz.get_num_components()

        if num_components == 1:
            self._shape = extents.y, extents.x
        else:
            self._shape = extents.y, extents.x, num_components

        self._is_planar = self._codestream.is_planar()
        self._num_components = num_components

        bit_depth = siz.get_bit_depth(0)
        is_signed = siz.is_signed(0)
        if bit_depth == 8 and not is_signed:
            self._dtype = np.uint8
        elif bit_depth == 8 and is_signed:
            self._dtype = np.int8
        elif bit_depth == 16 and not is_signed:
            self._dtype = np.uint16
        elif bit_depth == 16 and is_signed:
            self._dtype = np.int16
        elif bit_depth == 32 and not is_signed:
            self._dtype = np.uint32
        elif bit_depth == 32 and is_signed:
            self._dtype = np.int32
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}, signed: {is_signed}")

    @classmethod
    def from_memory(cls, data):
        """
        Create an OJPHImageFile instance from memory data.

        Parameters
        ----------
        data : numpy.ndarray
            The compressed JPEG2000 data as a numpy array.

        Returns
        -------
        OJPHImageFile
            An instance configured to read from the memory data.
        """
        instance = cls.__new__(cls)
        instance._codestream = None
        instance._ojph_file = None
        instance._filename = None

        ojph_file = MemInfile()
        ojph_file.open(data)
        instance._ojph_file = ojph_file
        instance._codestream = Codestream()
        instance._codestream.read_headers(ojph_file)

        siz = instance._codestream.access_siz()
        extents = siz.get_image_extent()
        num_components = siz.get_num_components()

        if num_components == 1:
            instance._shape = extents.y, extents.x
        else:
            instance._shape = extents.y, extents.x, num_components

        instance._is_planar = instance._codestream.is_planar()
        instance._num_components = num_components

        bit_depth = siz.get_bit_depth(0)
        is_signed = siz.is_signed(0)
        if bit_depth == 8 and not is_signed:
            instance._dtype = np.uint8
        elif bit_depth == 8 and is_signed:
            instance._dtype = np.int8
        elif bit_depth == 16 and not is_signed:
            instance._dtype = np.uint16
        elif bit_depth == 16 and is_signed:
            instance._dtype = np.int16
        elif bit_depth == 32 and not is_signed:
            instance._dtype = np.uint32
        elif bit_depth == 32 and is_signed:
            instance._dtype = np.int32
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}, signed: {is_signed}")

        return instance

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def _open_file(self):
        self._ojph_file = J2CInfile()
        self._ojph_file.open(self._filename)
        self._codestream = Codestream()
        self._codestream.read_headers(self._ojph_file)

    def read_image(self, *, level=0):
        if self._codestream is None:
            self._open_file()

        self._codestream.restrict_input_resolution(level, level)
        siz = self._codestream.access_siz()

        height = siz.get_recon_height(0)
        width = siz.get_recon_width(0)
        self._codestream.create()

        if self._num_components == 1:
            # Single component - always HW format
            image = np.zeros((height, width), dtype=self._dtype)

            for h in range(height):
                line = self._codestream.pull(0)
                i32_ptr = ctypes.cast(line.i32_address, ctypes.POINTER(ctypes.c_uint32))
                line_array = np.ctypeslib.as_array(
                    ctypes.cast(i32_ptr, ctypes.POINTER(ctypes.c_uint32)),
                    shape=(line.size,)
                )
                image[h] = line_array
        else:
            # Multi-component - optimize for RGB images using color transform detection
            is_rgb = self._codestream.access_cod().is_using_color_transform()

            if is_rgb:
                # RGB image - always return HWC format for optimal compatibility
                image = np.zeros((height, width, self._num_components), dtype=self._dtype)

                for c in range(self._num_components):
                    for h in range(height):
                        line = self._codestream.pull(c)
                        i32_ptr = ctypes.cast(line.i32_address, ctypes.POINTER(ctypes.c_uint32))
                        line_array = np.ctypeslib.as_array(
                            ctypes.cast(i32_ptr, ctypes.POINTER(ctypes.c_uint32)),
                            shape=(line.size,)
                        )
                        image[h, :, c] = line_array
            else:
                # Non-RGB multi-component - use planar flag for format detection
                if self._is_planar:
                    # Planar mode was used for writing - return CHW format
                    image = np.zeros((self._num_components, height, width), dtype=self._dtype)

                    for c in range(self._num_components):
                        for h in range(height):
                            line = self._codestream.pull(c)
                            i32_ptr = ctypes.cast(line.i32_address, ctypes.POINTER(ctypes.c_uint32))
                            line_array = np.ctypeslib.as_array(
                                ctypes.cast(i32_ptr, ctypes.POINTER(ctypes.c_uint32)),
                                shape=(line.size,)
                            )
                            image[c, h, :] = line_array
                else:
                    # Non-planar mode was used for writing - return HWC format
                    image = np.zeros((height, width, self._num_components), dtype=self._dtype)

                    for c in range(self._num_components):
                        for h in range(height):
                            line = self._codestream.pull(c)
                            i32_ptr = ctypes.cast(line.i32_address, ctypes.POINTER(ctypes.c_uint32))
                            line_array = np.ctypeslib.as_array(
                                ctypes.cast(i32_ptr, ctypes.POINTER(ctypes.c_uint32)),
                                shape=(line.size,)
                            )
                            image[h, :, c] = line_array

        self._close_codestream_and_file()
        return image

    def _close_codestream_and_file(self):
        if self._codestream is not None:
            self._codestream.close()
            # The codestream will close the infile automatically
        elif self._ojph_file is not None:
            self._ojph_file.close()
        self._codestream = None
        self._ojph_file = None

    def __del__(self):
        self._close_codestream_and_file()
