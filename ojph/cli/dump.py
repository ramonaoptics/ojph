import pathlib

import click

from ojph.ojph_bindings import Codestream, J2CInfile


def format_bool_as_int(value):
    return 1 if value else 0


def format_block_dims(block_dims):
    return f"cblkw=2^{block_dims.w}, cblkh=2^{block_dims.h}"


def format_precinct_sizes(cod_params, num_decompositions):
    sizes = []
    for level in range(num_decompositions + 1):
        size = cod_params.get_precinct_size(level)
        sizes.append(f"({size.w},{size.h})")
    return " ".join(sizes)


def parse_main_header_markers(path):
    entries = []
    with open(path, "rb") as f:
        pos = 0
        while True:
            marker_start = pos
            data = f.read(2)
            if len(data) < 2:
                break
            pos += 2
            marker = (data[0] << 8) | data[1]
            if marker == 0xFF90:
                break
            if marker == 0xFF4F:
                entries.append((marker, marker_start, 2))
                continue
            length_bytes = f.read(2)
            if len(length_bytes) < 2:
                break
            pos += 2
            length = (length_bytes[0] << 8) | length_bytes[1]
            entries.append((marker, marker_start, length))
            skip = length - 2
            if skip < 0:
                break
            f.seek(skip, 1)
            pos += skip
    return entries


def parse_tileparts(path):
    entries = []
    with open(path, "rb") as f:
        data = f.read()
    length = len(data)
    pos = 0
    while pos + 2 <= length:
        marker = (data[pos] << 8) | data[pos + 1]
        if marker == 0xFF4F:
            pos += 2
            continue
        if marker == 0xFF90:
            sot_pos = pos
            if pos + 12 > length:
                break
            lsot = (data[pos + 2] << 8) | data[pos + 3]
            if lsot != 10:
                break
            iset = (data[pos + 4] << 8) | data[pos + 5]
            psot = (
                (data[pos + 6] << 24)
                | (data[pos + 7] << 16)
                | (data[pos + 8] << 8)
                | data[pos + 9]
            )
            tpsot = data[pos + 10]
            tnsot = data[pos + 11]
            sod_pos = None
            scan_pos = pos + 2 + lsot
            while scan_pos + 2 <= length:
                if data[scan_pos] != 0xFF:
                    scan_pos += 1
                    continue
                code = (data[scan_pos] << 8) | data[scan_pos + 1]
                if code == 0xFF00:
                    scan_pos += 2
                    continue
                if code == 0xFF93:
                    sod_pos = scan_pos
                    break
                if code == 0xFF90 or code == 0xFFD9:
                    break
                if scan_pos + 4 > length:
                    break
                seg_len = (data[scan_pos + 2] << 8) | data[scan_pos + 3]
                if seg_len < 2:
                    break
                scan_pos += 2 + seg_len
            entries.append(
                {
                    "tile_index": iset,
                    "tile_part_index": tpsot,
                    "num_tile_parts": tnsot,
                    "sot_pos": sot_pos,
                    "sod_pos": sod_pos,
                    "psot": psot,
                }
            )
            pos += psot
            continue
        if marker == 0xFF93:
            pos += 2
            continue
        if marker == 0xFFD9:
            break
        if pos + 4 > length:
            break
        seg_len = (data[pos + 2] << 8) | data[pos + 3]
        if seg_len < 2:
            break
        pos += 2 + seg_len
    return entries


def dump_header(path, show_packets):
    infile = J2CInfile()
    infile.open(str(path))

    codestream = Codestream()
    codestream.read_headers(infile)

    siz = codestream.access_siz()
    cod = codestream.access_cod()

    image_offset = siz.get_image_offset()
    image_extent = siz.get_image_extent()
    tile_offset = siz.get_tile_offset()
    tile_size = siz.get_tile_size()
    num_components = siz.get_num_components()

    click.echo("Image info {")
    click.echo(f"\t x0={image_offset.x}, y0={image_offset.y}")
    click.echo(f"\t x1={image_extent.x}, y1={image_extent.y}")
    click.echo(f"\t numcomps={num_components}")
    for comp in range(num_components):
        downsampling = siz.get_downsampling(comp)
        bit_depth = siz.get_bit_depth(comp)
        is_signed = siz.is_signed(comp)
        click.echo(f"\t\t component {comp} {{")
        click.echo(f"\t\t dx={downsampling.x}, dy={downsampling.y}")
        click.echo(f"\t\t prec={bit_depth}")
        click.echo(f"\t\t sgnd={int(bool(is_signed))}")
        click.echo("\t}")
    click.echo("}")

    tile_width = tile_size.w
    tile_height = tile_size.h
    image_width = image_extent.x - image_offset.x
    image_height = image_extent.y - image_offset.y

    tiles_in_width = (image_width + tile_width - 1) // tile_width
    tiles_in_height = (image_height + tile_height - 1) // tile_height

    click.echo("Codestream info from main header: {")
    click.echo(f"\t tx0={tile_offset.x}, ty0={tile_offset.y}")
    click.echo(f"\t tdx={tile_width}, tdy={tile_height}")
    click.echo(f"\t tw={tiles_in_width}, th={tiles_in_height}")
    click.echo("\t default tile {")

    progression_order = cod.get_progression_order()
    num_layers = cod.get_num_layers()
    uses_color_transform = cod.is_using_color_transform()
    num_decompositions = cod.get_num_decompositions()
    block_dims = cod.get_log_block_dims()

    click.echo("\t\t csty=0")
    click.echo(f"\t\t prg=0x{progression_order:x}")
    click.echo(f"\t\t numlayers={num_layers}")
    click.echo(f"\t\t mct={format_bool_as_int(uses_color_transform)}")

    for comp in range(num_components):
        click.echo(f"\t\t comp {comp} {{")
        click.echo("\t\t\t csty=0")
        click.echo(f"\t\t\t numresolutions={num_decompositions + 1}")
        click.echo(f"\t\t\t {format_block_dims(block_dims)}")
        click.echo("\t\t\t cblksty=0x0")
        click.echo("\t\t\t qmfbid=1")
        precinct_string = format_precinct_sizes(cod, num_decompositions)
        click.echo(f"\t\t\t preccintsize (w,h)={precinct_string}")
        click.echo("\t\t\t qntsty=0")
        click.echo("\t\t\t numgbits=0")
        click.echo("\t\t\t roishift=0")
        click.echo("\t\t }")
    click.echo("\t }")
    click.echo("}")

    markers = parse_main_header_markers(str(path))
    if markers:
        start_pos = markers[0][1]
        end_pos = markers[-1][1] + markers[-1][2]
        click.echo("Codestream index from main header: {")
        click.echo(f"\t Main header start position={start_pos}")
        click.echo(f"\t Main header end position={end_pos}")
        click.echo("\t Marker list: {")
        for marker, pos, length in markers:
            click.echo(f"\t\t type=0x{marker:04x}, pos={pos}, len={length}")
        click.echo("\t }")
        click.echo("}")

    if show_packets:
        tileparts = parse_tileparts(str(path))
        if tileparts:
            click.echo("Tile-parts {")
            for entry in tileparts:
                click.echo(
                    f"\t tile={entry['tile_index']} "
                    f"tilepart={entry['tile_part_index'] + 1}/{entry['num_tile_parts']} "
                    f"sot_pos={entry['sot_pos']} "
                    f"sod_pos={entry['sod_pos']} "
                    f"length={entry['psot']}"
                )
            click.echo("}")

    codestream.close()


@click.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option("--show-packets", is_flag=True, help="Show tile-part offsets")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
def main(input, show_packets, verbose):
    dump_header(input, show_packets)
