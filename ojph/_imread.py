import numpy as np
from warnings import warn

from .ojph_bindings import J2CInfile, MemInfile, Codestream

def imread(
    uri,
    *,
    index=None,
    plugin=None,
    extension=None,
    format_hint=None,
    channel_order=None,
    offset=None,
    level=0,
    skipped_res_for_data=None,
    skipped_res_for_recon=None,
    **kwargs,
):
    if index is not None:
        warn(f"index {index} is ignored", stacklevel=2)
    if plugin is not None:
        warn(f"plugin {plugin} is ignored", stacklevel=2)
    if extension is not None:
        warn(f"extension {extension} is ignored", stacklevel=2)
    if format_hint is not None:
        warn(f"format_hint {format_hint} is ignored", stacklevel=2)

    return OJPHImageFile(uri, channel_order=channel_order, offset=None).read_image(
        level=level,
        skipped_res_for_data=skipped_res_for_data,
        skipped_res_for_recon=skipped_res_for_recon,
    )


def imread_from_memory(
    data,
    *,
    channel_order=None,
    level=0,
    skipped_res_for_data=None,
    skipped_res_for_recon=None,
    out=None,
):
    """Read a JPEG2000 image from memory data.

    Parameters
    ----------
    data : numpy.ndarray or bytes-like
        The compressed JPEG2000 data as a numpy array or bytes-like object.
    channel_order : str, optional
        Channel order specification ('HWC' or 'CHW').
    level : int, optional
        Resolution level to read (default: 0 for full resolution).
    skipped_res_for_data : int, optional
        Number of fine resolutions to skip during decoding.
    skipped_res_for_recon : int, optional
        Number of fine resolutions to skip during reconstruction.

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

    return OJPHImageFile.from_memory(data, channel_order=channel_order).read_image(
        level=level,
        skipped_res_for_data=skipped_res_for_data,
        skipped_res_for_recon=skipped_res_for_recon,
        out=out,
    )


class OJPHImageFile:
    def __init__(self, filename, *, mode='r', channel_order=None, offset=None):
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
            if offset is not None:
                ojph_file.seek(offset, 0)
            self._ojph_file = ojph_file

        self._codestream = Codestream()
        self._codestream.read_headers(self._ojph_file)

        siz = self._codestream.access_siz()
        extents = siz.get_image_extent()
        num_components = siz.get_num_components()

        self._is_planar = self._codestream.is_planar()
        self._num_components = num_components
        self._channel_order = channel_order

        cod = self._codestream.access_cod()
        self._num_decompositions = cod.get_num_decompositions()
        self._progression_order = cod.get_progression_order_as_string()

        if self._channel_order is None:
            if cod.is_using_color_transform():
                self._channel_order = 'HWC'
            else:
                self._channel_order = 'HWC' if self._is_planar else 'CHW'

        self._shape = extents.y, extents.x
        if self._num_components > 1:
            if self._channel_order == "HWC":
                self._shape = self._shape + (self._num_components,)
            else:
                self._shape = (self._num_components,) + self._shape

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
    def from_memory(cls, data, *, channel_order=None, offset=None):
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
        instance._channel_order = channel_order
        instance._codestream = None
        instance._ojph_file = None
        instance._filename = None

        ojph_file = MemInfile()
        ojph_file.open(data)
        if offset is not None:
            ojph_file.seek(offset, 0)
        instance._ojph_file = ojph_file
        instance._codestream = Codestream()
        instance._codestream.read_headers(ojph_file)

        siz = instance._codestream.access_siz()
        extents = siz.get_image_extent()
        num_components = siz.get_num_components()

        instance._is_planar = instance._codestream.is_planar()
        instance._num_components = num_components
        instance._channel_order = channel_order

        cod = instance._codestream.access_cod()
        instance._num_decompositions = cod.get_num_decompositions()
        instance._progression_order = cod.get_progression_order_as_string()

        if instance._channel_order is None:
            if cod.is_using_color_transform():
                instance._channel_order = 'HWC'
            else:
                instance._channel_order = 'HWC' if instance._is_planar else 'CHW'

        instance._shape = extents.y, extents.x
        if instance._num_components > 1:
            if instance._channel_order == "HWC":
                instance._shape = instance._shape + (instance._num_components,)
            else:
                instance._shape = (instance._num_components,) + instance._shape

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

    @property
    def levels(self):
        return self._num_decompositions

    @property
    def progression_order(self):
        return self._progression_order

    def _open_file(self):
        self._ojph_file = J2CInfile()
        self._ojph_file.open(self._filename)
        self._codestream = Codestream()
        self._codestream.read_headers(self._ojph_file)

    def read_image(
        self,
        *,
        level=0,
        skipped_res_for_data=None,
        skipped_res_for_recon=None,
        out=None,
    ):
        if self._codestream is None:
            self._open_file()

        if skipped_res_for_data is None:
            skipped_res_for_data = level
        if skipped_res_for_recon is None:
            skipped_res_for_recon = level

        if skipped_res_for_recon > self._num_decompositions:
            raise ValueError(
                f"skipped_res_for_recon ({skipped_res_for_recon}) "
                f"cannot be greater than the number of decompositions ({self._num_decompositions})"
            )
        if skipped_res_for_data > self._num_decompositions:
            raise ValueError(
                f"skipped_res_for_data ({skipped_res_for_data}) "
                f"cannot be greater than the number of decompositions ({self._num_decompositions})"
            )

        self._codestream.restrict_input_resolution(
            skipped_res_for_data,
            skipped_res_for_recon,
        )
        siz = self._codestream.access_siz()

        height = siz.get_recon_height(0)
        width = siz.get_recon_width(0)
        self._codestream.create()

        if self._num_components == 1:
            shape = (height, width)
        elif self._channel_order == 'CHW':
            shape = (self._num_components, height, width)
        else:
            shape = (height, width, self._num_components)

        if out is None:
            image = np.zeros(shape, dtype=self._dtype)
        else:
            if out.dtype != self._dtype:
                raise ValueError(
                    f"dtype mismatch. out was provided with {out.dtype} but it must be {self._dtype}"
                )
            # Potentially collapse any additional dimensions
            # do not use reshape since reshape can return a copy
            image = out.view()
            image.shape = shape


        if self._dtype in [np.uint32, np.int32]:
            min_val = None
            max_val = None
        else:
            iinfo = np.iinfo(self._dtype)
            min_val = iinfo.min
            max_val = iinfo.max

        self._codestream.pull_all_components(image, self._num_components, self._channel_order, min_val, max_val)

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
