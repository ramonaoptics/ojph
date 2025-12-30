import numpy as np
import inspect
from collections.abc import Buffer

from .ojph_bindings import Codestream, J2COutfile, MemOutfile, Point


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


def imwrite_to_memory(
    image,
    *,
    channel_order=None,
    num_decompositions=None,
    reversible=None,
    qstep=None,
    progression_order=None,
    tlm_marker=True,
    tileparts_at_resolutions=None,
    tileparts_at_components=None,
):
    mem_outfile = MemOutfile()
    mem_outfile.open(65536, False)
    codestream = Codestream()
    imwrite(
        mem_outfile,
        image,
        channel_order=channel_order,
        codestream=codestream,
        num_decompositions=num_decompositions,
        reversible=reversible,
        qstep=qstep,
        progression_order=progression_order,
        tlm_marker=tlm_marker,
        tileparts_at_resolutions=tileparts_at_resolutions,
        tileparts_at_components=tileparts_at_components,
    )
    return np.asarray(CompressedData(mem_outfile, codestream))


def imwrite(
    filename,
    image,
    *,
    channel_order=None,
    codestream=None,
    num_decompositions=None,
    reversible=None,
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
    if reversible is None:
        reversible = True
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

    codestream.push_all_components(image, num_components, channel_order)

    codestream.flush()
    if close_codestream:
        codestream.close()
