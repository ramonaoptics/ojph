#!/usr/bin/env python3
import click
import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile, COMPRESSION
from ojph._imread import OJPHImageFile


@click.command()
@click.argument('filename', type=click.Path(exists=True))
@click.option('--index', '-i', type=int, default=0, help='Page index to display')
@click.option('--list-pages', '-l', is_flag=True, help='List all available pages')
@click.option('--level', type=int, default=0, help='Resolution level to read (0 = full resolution)')
@click.option('--skipped-res-for-data', type=int, default=None, help='Number of fine resolutions to skip during decoding')
@click.option('--skipped-res-for-recon', type=int, default=None, help='Number of fine resolutions to skip during reconstruction')
def main(filename, index, list_pages, level, skipped_res_for_data, skipped_res_for_recon):
    tif = TiffFile(filename)
    num_pages = len(tif.pages)

    if list_pages:
        click.echo(f"Found {num_pages} pages in {filename}")
        for i, page in enumerate(tif.pages):
            compression = page.compression
            if compression == COMPRESSION.JPEG2000:
                compression_name = "JPEG2000"
            else:
                compression_name = f"Other ({compression})"
            click.echo(f"  Page {i}: {compression_name}")
        tif.close()
        return

    if index < 0 or index >= num_pages:
        click.echo(f"Error: Index {index} is out of range. File has {num_pages} pages (0-{num_pages-1})", err=True)
        tif.close()
        return

    page = tif.pages[index]

    if page.compression != COMPRESSION.JPEG2000:
        click.echo(f"Error: Page {index} is not JPEG2000 compressed (compression: {page.compression})", err=True)
        tif.close()
        return

    offset = page.offset
    tif.close()

    from pathlib import Path
    name = Path(filename).name
    if skipped_res_for_data is not None or skipped_res_for_recon is not None:
        click.echo(f"Reading page {index} from {name} with offset {offset} (skipped_res_for_data={skipped_res_for_data}, skipped_res_for_recon={skipped_res_for_recon})")
    else:
        click.echo(f"Reading page {index} from {name} with offset {offset} at level {level}")

    try:
        image_file = OJPHImageFile(filename, offset=offset)
        image = image_file.read_image(
            level=level,
            skipped_res_for_data=skipped_res_for_data,
            skipped_res_for_recon=skipped_res_for_recon,
        )

        click.echo(f"Image shape: {image.shape}, dtype: {image.dtype}")

        plt.figure(figsize=(10, 10))
        if image.ndim == 2:
            plt.imshow(image, cmap='gray')
        elif image.ndim == 3:
            plt.imshow(image)
        else:
            click.echo(f"Error: Unexpected image dimensions: {image.ndim}", err=True)
            return

        if skipped_res_for_data is not None or skipped_res_for_recon is not None:
            title = f"Page {index} from {name} (data={skipped_res_for_data}, recon={skipped_res_for_recon})"
        else:
            title = f"Page {index} from {name} (level {level})"
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        click.echo(f"Error reading image: {e}", err=True)
        raise


if __name__ == '__main__':
    main()
