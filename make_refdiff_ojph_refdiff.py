#!/usr/bin/env python
"""
OpenJPH r1x1 (identity wavelet) demo: circle + LL vs subsample comparison.

With r1x1=True, the wavelet is identity (L=even, H=odd), so LL at each level
equals 2x2 subsampling and the diff images should be zero.
"""

from pathlib import Path

import numpy as np
import imageio

from ojph import imread, imwrite


def make_circle(h=1024, w=1024, radius_fraction=0.25):
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * radius_fraction)
    img = np.zeros((h, w), dtype=np.uint8)
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    img[mask] = 255
    return img


def main():
    out_dir = Path("./refdiff_demo")
    out_dir.mkdir(exist_ok=True)

    base = make_circle()

    pgm_path = out_dir / "ojph_r1x1_circle_1024x1024.pgm"
    imageio.imwrite(pgm_path, base)
    print(f"Wrote original pgm: {pgm_path}")

    j2c_path = out_dir / "ojph_r1x1.j2c"
    imwrite(
        str(j2c_path),
        base,
        reversible=True,
        r1x1=True,
    )
    print(f"Wrote r1x1 codestream: {j2c_path}")

    roundtrip = imread(str(j2c_path))
    roundtrip_pgm = out_dir / "ojph_r1x1_roundtrip.pgm"
    imageio.imwrite(roundtrip_pgm, roundtrip)
    print(f"Wrote full-res roundtrip: {roundtrip_pgm}")

    print("Computing LL vs subsample images via ojph imread(level=...) and NumPy...")
    for level in (1, 2, 3, 4):
        ll = imread(str(j2c_path), level=level)

        step = 2**level
        subsampled = base[::step, ::step]

        if ll.shape != subsampled.shape:
            print(f"Level {level}: shape mismatch {ll.shape} vs {subsampled.shape}")
        else:
            diff = ll.astype(np.int16) - subsampled.astype(np.int16)
            num_diff = np.count_nonzero(diff)
            print(f"Level {level}: LL vs subsample differ at {num_diff} pixels")

        diff = ll.astype(np.int16) - subsampled.astype(np.int16)
        diff_vis = np.clip(diff + 128, 0, 255).astype(np.uint8)

        ll_path = out_dir / f"ojph_r1x1_ll_codec_L{level}.pgm"
        sub_path = out_dir / f"ojph_r1x1_ll_subsample_L{level}.pgm"
        diff_path = out_dir / f"ojph_r1x1_ll_diff_L{level}.pgm"

        imageio.imwrite(ll_path, ll.astype(np.uint8))
        imageio.imwrite(sub_path, subsampled.astype(np.uint8))
        imageio.imwrite(diff_path, diff_vis)

        print(f"  Wrote codec LL              -> {ll_path}")
        print(f"  Wrote simple subsample      -> {sub_path}")
        print(f"  Wrote centered diff (+128)  -> {diff_path}")

    print("\nDone. View files in './refdiff_demo' for r1x1 (identity) kernel comparison.")


if __name__ == "__main__":
    main()
