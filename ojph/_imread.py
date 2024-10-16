import os
import numpy as np
import ctypes
from .ojph_bindings import J2CInfile, Codestream
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


class OJPHImageFile:
    def __init__(self, filename):
        self._codestream = None
        self._ojph_file = None
        self._filename = filename

        self._ojph_file = J2CInfile()
        self._ojph_file.open(str(filename))
        self._codestream = Codestream()
        self._codestream.read_headers(self._ojph_file)


        siz = self._codestream.access_siz()
        extents = siz.get_image_extent()
        self._shape = extents.y, extents.x
        self._is_planar = self._codestream.is_planar()

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
            raise ValueError("Unsupported bit depth")

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

        image = np.zeros(
            (height, width),
            dtype=self._dtype
        )

        for h in range(height):
            line = self._codestream.pull(0)
            # Convert the address to a ctypes pointer to int32
            i32_ptr = ctypes.cast(line.i32_address, ctypes.POINTER(ctypes.c_uint32))

            # Calculate the total number of bytes (size of the array in elements * size of int32)
            line_array = np.ctypeslib.as_array(
                ctypes.cast(i32_ptr, ctypes.POINTER(ctypes.c_uint32)),
                shape=(line.size,)
            )
            image[h] = line_array

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