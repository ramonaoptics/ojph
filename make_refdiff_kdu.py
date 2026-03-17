#!/usr/bin/env python

import subprocess
from pathlib import Path

import numpy as np
import imageio


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

    pgm_path = out_dir / "kdu_circle_1024x1024.pgm"
    imageio.imwrite(pgm_path, base)
    print(f"Wrote original pgm: {pgm_path}")

    j2c_path = out_dir / "kdu_rev53.j2c"
    cmd_compress = [
        "kdu_compress",
        "-i",
        str(pgm_path),
        "-o",
        str(j2c_path),
        "-rate",
        "1.0",
        "Creversible=yes",
    ]
    print("Running:", " ".join(cmd_compress))
    subprocess.run(cmd_compress, check=True)

    roundtrip_pgm = out_dir / "kdu_rev53_roundtrip.pgm"
    cmd_expand_full = [
        "kdu_expand",
        "-i",
        str(j2c_path),
        "-o",
        str(roundtrip_pgm),
    ]
    print("Running:", " ".join(cmd_expand_full))
    subprocess.run(cmd_expand_full, check=True)

    print("Computing LL vs subsample images via kdu_expand and NumPy...")
    for level in (1, 2, 3, 4):
        ll_path = out_dir / f"kdu_ll_codec_L{level}.pgm"
        cmd_ll = [
            "kdu_expand",
            "-i",
            str(j2c_path),
            "-o",
            str(ll_path),
            "-layers",
            "1",
            "-reduce",
            str(level),
        ]
        print("Running:", " ".join(cmd_ll))
        subprocess.run(cmd_ll, check=True)

        ll = imageio.imread(ll_path)

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

        sub_path = out_dir / f"kdu_ll_subsample_L{level}.pgm"
        diff_path = out_dir / f"kdu_ll_diff_L{level}.pgm"

        imageio.imwrite(sub_path, subsampled.astype(np.uint8))
        imageio.imwrite(diff_path, diff_vis)

        print(f"  Wrote codec LL              -> {ll_path}")
        print(f"  Wrote simple subsample      -> {sub_path}")
        print(f"  Wrote centered diff (+128)  -> {diff_path}")

    print("\nDone. View files in './refdiff_demo' to see differences.")


if __name__ == "__main__":
    main()

