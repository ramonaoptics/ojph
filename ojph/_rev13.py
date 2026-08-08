"""Main-header fixups that declare the rev13 wavelet kernel.

``Codestream.install_rev13_wavelet()`` changes the kernel the *encoder* runs,
but the codestream it produces still describes itself as an ordinary reversible
5/3 one, because everything OpenJPH writes into the main header is decided
before a single sample is transformed.  Rewriting the header afterwards is what
makes the two agree, and it is the whole reason rev13 works against a stock
OpenJPH -- see the long comment in ``ojph/ojph_bindings.cpp``.

Three edits, all confined to the main header:

* ``SIZ-Rsiz`` gains the Part 2 extension flags, since an ATK-signalled kernel
  is not a Part 1 feature.
* ``COD-SPcod.wavelet_trans`` goes from 1 (the 5/3 kernel, implied by Part 1)
  to 2, the ATK index rev13 is signalled with.
* an ATK marker segment describing rev13 is spliced in immediately before COD,
  the same position a patched OpenJPH writes it at.

Nothing outside the main header moves, and no marker segment carries absolute
offsets (TLM records tile-part *lengths*), so inserting bytes here cannot
invalidate anything downstream.
"""
from .ojph_bindings import REV13_WAVELET_INDEX, rev13_atk_marker_segment

# Markers, as they appear in the codestream.
_SOC = 0xFF4F  # start of codestream
_SIZ = 0xFF51  # image and tile size
_COD = 0xFF52  # coding style default
_SOT = 0xFF90  # start of tile-part -- ends the main header

# Rsiz flags: Part 2 extensions in use, and specifically a Part 2 wavelet kernel
# signalled by an ATK marker segment.
_RSIZ_EXT_FLAG = 0x8000
_RSIZ_WS_KERN_FLAG = 0x0020

# Offset of SPcod.wavelet_trans from the start of the COD marker segment:
# COD(2) Lcod(2) Scod(1) prog_order(1) num_layers(2) mc_trans(1) num_decomp(1)
# block_width(1) block_height(1) block_style(1), then wavelet_trans.
_COD_WAVELET_TRANS_OFFSET = 13

# The Part 1 reversible 5/3 kernel, which is what the encoder claims to have
# used before we rewrite the header.
_DWT_REV53 = 1


def _iter_main_header_segments(data):
    """Yield ``(marker, offset, length)`` for each main-header marker segment.

    ``offset`` is the position of the marker itself and ``length`` spans the
    marker plus its parameters.  Stops at SOT, the first tile-part.
    """
    if len(data) < 4 or int.from_bytes(data[0:2], 'big') != _SOC:
        raise ValueError("not a JPEG 2000 codestream: no SOC marker")
    pos = 2
    while pos + 4 <= len(data):
        marker = int.from_bytes(data[pos:pos + 2], 'big')
        if marker == _SOT:
            return
        if marker < 0xFF01 or marker > 0xFFFE:
            raise ValueError(
                f"malformed main header: expected a marker at byte {pos}, "
                f"found 0x{marker:04X}"
            )
        segment_length = int.from_bytes(data[pos + 2:pos + 4], 'big')
        if segment_length < 2 or pos + 2 + segment_length > len(data):
            raise ValueError(
                f"malformed marker segment 0x{marker:04X} at byte {pos}"
            )
        yield marker, pos, 2 + segment_length
        pos += 2 + segment_length
    raise ValueError("codestream ended before the first tile-part (SOT)")


def declare_rev13_wavelet(data):
    """Return ``data`` with its main header rewritten to declare rev13.

    ``data`` is a codestream that was encoded with the rev13 kernel but whose
    header still describes the 5/3 kernel.
    """
    data = bytearray(data)

    siz_offset = None
    cod_offset = None
    for marker, offset, _ in _iter_main_header_segments(data):
        if marker == _SIZ and siz_offset is None:
            siz_offset = offset
        elif marker == _COD and cod_offset is None:
            cod_offset = offset
    if siz_offset is None:
        raise ValueError("codestream has no SIZ marker segment")
    if cod_offset is None:
        raise ValueError("codestream has no COD marker segment")

    # SIZ-Rsiz sits right after the marker and its length field.
    rsiz_offset = siz_offset + 4
    rsiz = int.from_bytes(data[rsiz_offset:rsiz_offset + 2], 'big')
    rsiz |= _RSIZ_EXT_FLAG | _RSIZ_WS_KERN_FLAG
    data[rsiz_offset:rsiz_offset + 2] = rsiz.to_bytes(2, 'big')

    wavelet_offset = cod_offset + _COD_WAVELET_TRANS_OFFSET
    if data[wavelet_offset] != _DWT_REV53:
        raise ValueError(
            "expected the codestream to have been encoded with the "
            f"reversible 5/3 kernel ({_DWT_REV53}), found kernel "
            f"{data[wavelet_offset]}"
        )
    data[wavelet_offset] = REV13_WAVELET_INDEX

    # The ATK marker segment goes immediately before COD, where a patched
    # OpenJPH writes it.
    atk = rev13_atk_marker_segment()
    return bytes(data[:cod_offset]) + atk + bytes(data[cod_offset:])
