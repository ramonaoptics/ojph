import numpy as np
import inspect
from collections.abc import Buffer

from ._rev13 import declare_rev13_wavelet
from .ojph_bindings import Codestream, J2COutfile, MemOutfile, Point

# The wavelet kernels ``imwrite`` accepts, and whether each one is reversible.
# 'irv97' and 'rev53' are the Part 1 kernels, which ``reversible=`` has always
# selected between. 'rev13' is the reversible predict-only kernel: every
# decomposition's low-pass subband is the even-indexed samples of the previous
# resolution, untouched, so decoding ``level=r`` returns exactly
# ``image[::2**r, ::2**r]``.
_VALID_WAVELETS = {'irv97': False, 'rev53': True, 'rev13': True}


class CompressedData(Buffer):
    def __init__(self, mem_file, codestream):
        self._mem_file = mem_file
        self._codestream = codestream
        self._memoryview = None

    def __del__(self):
        """Clean up the memory file when this object is garbage collected."""
        if self._codestream is not None:
            self._codestream.close()
        self._codestream = None
        if self._mem_file is not None:
            self._mem_file.close()
        self._mem_file = None
        self._memoryview = None

    def __buffer__(self, flags: int) -> Buffer:
        if flags & inspect.BufferFlags.WRITABLE:
            raise TypeError("CompressedData is read-only")
        if self._memoryview is None:
            self._memoryview = self._mem_file.get_data()
        return self._memoryview


def _encode_to_bytes(image, *, codestream=None, wavelet=None, **kwargs):
    """Encode into memory and return the finished codestream as bytes."""
    mem_outfile = MemOutfile()
    mem_outfile.open(65536, False)
    close_codestream = codestream is None
    if close_codestream:
        codestream = Codestream()
    imwrite(
        mem_outfile,
        image,
        codestream=codestream,
        wavelet=wavelet,
        **kwargs,
    )
    data = bytes(mem_outfile.get_data())
    if close_codestream:
        codestream.close()
    mem_outfile.close()
    if wavelet is not None and wavelet.lower() == 'rev13':
        # The encoder ran rev13, but the header it wrote still says 5/3.
        data = declare_rev13_wavelet(data)
    return data


def imwrite_to_memory(
    image,
    *,
    channel_order=None,
    num_decompositions=None,
    reversible=None,
    wavelet=None,
    qstep=None,
    progression_order=None,
    tlm_marker=True,
    tileparts_at_resolutions=None,
    tileparts_at_components=None,
):
    data = _encode_to_bytes(
        image,
        channel_order=channel_order,
        num_decompositions=num_decompositions,
        reversible=reversible,
        wavelet=wavelet,
        qstep=qstep,
        progression_order=progression_order,
        tlm_marker=tlm_marker,
        tileparts_at_resolutions=tileparts_at_resolutions,
        tileparts_at_components=tileparts_at_components,
    )
    return np.frombuffer(data, dtype=np.uint8)


def imwrite(
    filename,
    image,
    *,
    channel_order=None,
    codestream=None,
    num_decompositions=None,
    reversible=None,
    wavelet=None,
    qstep=None,
    progression_order=None,
    tlm_marker=True,
    tileparts_at_resolutions=None,
    tileparts_at_components=None,
):
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

    if wavelet is not None:
        wavelet = wavelet.lower()
        if wavelet not in _VALID_WAVELETS:
            raise ValueError(
                f"Invalid wavelet '{wavelet}'. "
                f"Must be one of: {', '.join(sorted(_VALID_WAVELETS))}"
            )
        if reversible is None:
            reversible = _VALID_WAVELETS[wavelet]
        elif reversible != _VALID_WAVELETS[wavelet]:
            raise ValueError(
                f"The wavelet '{wavelet}' is "
                f"{'reversible' if _VALID_WAVELETS[wavelet] else 'irreversible'}"
                f", which contradicts reversible={reversible}."
            )
    if reversible is None:
        reversible = True

    if wavelet == 'rev13' and not isinstance(filename, MemOutfile):
        # rev13 declares itself through main-header edits that can only be made
        # once the codestream is complete (see ojph/_rev13.py), so build it in
        # memory and write the finished bytes out. Passing a MemOutfile in
        # directly encodes straight into it and skips that step, which is how
        # _encode_to_bytes drives this function.
        data = _encode_to_bytes(
            image,
            channel_order=channel_order,
            codestream=codestream,
            num_decompositions=num_decompositions,
            reversible=reversible,
            wavelet=wavelet,
            qstep=qstep,
            progression_order=progression_order,
            tlm_marker=tlm_marker,
            tileparts_at_resolutions=tileparts_at_resolutions,
            tileparts_at_components=tileparts_at_components,
        )
        with open(filename, 'wb') as f:
            f.write(data)
        return

    if isinstance(filename, MemOutfile):
        ojph_file = filename
    else:
        ojph_file = J2COutfile()
        ojph_file.open(str(filename))

    close_codestream = codestream is None
    if codestream is None:
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
    if progression_order is None:
        progression_order = "RLCP"

    progression_order = progression_order.upper()
    valid_progressions = {"LRCP", "RLCP", "RPCL", "PCRL", "CPRL"}
    if progression_order not in valid_progressions:
        raise ValueError(
            f"Invalid progression_order '{progression_order}'. "
            f"Must be one of: {', '.join(sorted(valid_progressions))}"
        )
    cod.set_progression_order(progression_order)
    cod.set_reversible(reversible)
    cod.set_color_transform(False)
    if num_decompositions is not None:
        cod.set_num_decomposition(num_decompositions)
    if not reversible and qstep is not None:
        codestream.access_qcd().set_irrev_quant(qstep)
    codestream.set_planar(num_components > 1)
    if tileparts_at_resolutions is None:
        tileparts_at_resolutions = progression_order == "RLCP"
    if tileparts_at_components is None:
        tileparts_at_components = False
    codestream.set_tilepart_divisions(tileparts_at_resolutions, tileparts_at_components)
    codestream.request_tlm_marker(tlm_marker)

    codestream.write_headers(ojph_file, None, 0)
    if wavelet == 'rev13':
        # Swap the 5/3 analysis kernel the header was written for out for
        # rev13. write_headers() is what linked the kernel to COD, and no
        # sample has been transformed yet, so this is the one window in which
        # it can be done.
        codestream.install_rev13_wavelet()

    # For native byte orders, even if the byte order of the input is
    # explicitely set
    # this will be a no-operation
    # and helps streamline code inside push_all_components
    if image.dtype.byteorder not in ("=", "|"):
        image = np.asarray(image,  dtype=image.dtype.newbyteorder('='))
    codestream.push_all_components(image, num_components, channel_order)

    codestream.flush()
    if close_codestream:
        codestream.close()
