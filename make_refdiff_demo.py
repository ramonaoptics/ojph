#!/usr/bin/env python

import subprocess
from pathlib import Path

import numpy as np
import imageio


def main():
    out_dir = Path("./refdiff_demo")
    out_dir.mkdir(exist_ok=True)

    # 1. Generate base grayscale image: binary circle centered in 1024x1024
    h, w = 1024, 1024
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    base = np.zeros((h, w), dtype=np.uint8)
    base[(x - cx) ** 2 + (y - cy) ** 2 <= r * r] = 255

    pgm_path = out_dir / "seed0_1024x1024.pgm"
    imageio.imwrite(pgm_path, base)
    print(f"Wrote original pgm: {pgm_path}")

    # 2. Compress with reversible 5/3 using ojph_compress
    j2c_path = out_dir / "seed0_rev53.j2c"
    cmd_compress = [
        "ojph_compress",
        "-i", str(pgm_path),
        "-o", str(j2c_path),
        "-reversible", "true",
    ]
    print("Running:", " ".join(cmd_compress))
    subprocess.run(cmd_compress, check=True)

    # 3. Expand full-res image with ojph_expand (for sanity)
    roundtrip_pgm = out_dir / "seed0_rev53_roundtrip.pgm"
    cmd_expand = [
        "ojph_expand",
        "-i", str(j2c_path),
        "-o", str(roundtrip_pgm),
    ]
    print("Running:", " ".join(cmd_expand))
    subprocess.run(cmd_expand, check=True)

    # 4. Extract reduced-resolution images for several levels using ojph_expand
    #    and create comparison and diff images (no Python bindings).
    print("Computing LL vs subsample images via ojph_expand and NumPy...")
    for level in (1, 2, 3, 4):
        # Use ojph_expand with -skip_res L,L to reconstruct LL at this level
        ll_path = out_dir / f"seed0_ll_codec_L{level}.pgm"
        cmd_ll = [
            "ojph_expand",
            "-i", str(j2c_path),
            "-o", str(ll_path),
            "-skip_res", f"{level},{level}",
        ]
        print("Running:", " ".join(cmd_ll))
        subprocess.run(cmd_ll, check=True)

        ll = imageio.imread(ll_path)

        step = 2 ** level
        subsampled = base[::step, ::step]

        if ll.shape != subsampled.shape:
            print(f"Level {level}: shape mismatch {ll.shape} vs {subsampled.shape}")
        else:
            diff = ll.astype(np.int16) - subsampled.astype(np.int16)
            num_diff = np.count_nonzero(diff)
            print(f"Level {level}: LL vs subsample differ at {num_diff} pixels")

        # Visualization: center differences at 128 to show negatives/positives
        diff = ll.astype(np.int16) - subsampled.astype(np.int16)
        diff_vis = np.clip(diff + 128, 0, 255).astype(np.uint8)

        sub_path = out_dir / f"seed0_ll_subsample_L{level}.pgm"
        diff_path = out_dir / f"seed0_ll_diff_L{level}.pgm"

        imageio.imwrite(sub_path, subsampled.astype(np.uint8))
        imageio.imwrite(diff_path, diff_vis)

        print(f"  Wrote codec LL              -> {ll_path}")
        print(f"  Wrote simple subsample      -> {sub_path}")
        print(f"  Wrote centered diff (+128)  -> {diff_path}")

    print("\nDone. View files in './refdiff_demo' to see differences.")


if __name__ == "__main__":
    main()
