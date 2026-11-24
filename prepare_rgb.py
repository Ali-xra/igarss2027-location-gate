# -*- coding: utf-8 -*-
"""
prepare_rgb.py - build the RGB version of eurosat_s1_cache.npz, row-aligned with it.

Raw data: EuroSAT RGB (Helber et al., MIT licence), the archive `EuroSAT_RGB.zip`
(download links: https://github.com/phelber/EuroSAT). Run prepare_s1.py first.

EuroSAT-S1 was built from the same 27,000 EuroSAT chips, and the Sentinel-1 cache keeps
their file names. This script writes the RGB images in exactly the row order of the
Sentinel-1 cache, so row i is the same chip in both files. Coordinates, labels and the
Copernicus-Bench split are copied from the Sentinel-1 cache; only the pixels change.

Writes (default ./data/eurosat_rgb_cache.npz): images (N, 3, 64, 64) float32 in [0, 1],
coords, labels, splits, names, classes.

    python prepare_rgb.py --zip path/to/EuroSAT_RGB.zip
"""
import argparse, sys, zipfile, io
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="EuroSAT_RGB.zip")
    ap.add_argument("--s1-cache", default="./data/eurosat_s1_cache.npz")
    ap.add_argument("--out", default="./data/eurosat_rgb_cache.npz")
    a = ap.parse_args()
    out_f = Path(a.out)
    out_f.parent.mkdir(parents=True, exist_ok=True)

    s1 = np.load(a.s1_cache, allow_pickle=True)
    names = [str(x) for x in s1["names"]]
    print(f"Sentinel-1 cache: {len(names)} chips, first = {names[0]}", flush=True)

    zf = zipfile.ZipFile(a.zip)
    members = {}
    for m in zf.namelist():
        if m.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
            members[Path(m).stem] = m
    print(f"archive: {len(members)} image members", flush=True)

    missing = [n for n in names if Path(n).stem not in members]
    print(f"   chips in the Sentinel-1 cache with no RGB twin: {len(missing)}", flush=True)
    if missing[:5]:
        print(f"   e.g. {missing[:5]}", flush=True)
    assert len(missing) == 0, "the two caches do not cover the same chips -- stop"

    probe = np.array(Image.open(io.BytesIO(zf.read(members[Path(names[0]).stem]))))
    print(f"   RGB image shape {probe.shape} dtype {probe.dtype}", flush=True)
    H, W = probe.shape[:2]
    C = probe.shape[2] if probe.ndim == 3 else 1

    imgs = np.empty((len(names), C, H, W), dtype=np.float32)
    for i, n in enumerate(names):
        arr = np.array(Image.open(io.BytesIO(zf.read(members[Path(n).stem]))), dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        imgs[i] = arr.transpose(2, 0, 1)
        if i % 4000 == 0:
            print(f"      {i}/{len(names)}", flush=True)

    # scale to [0, 1]; train.py standardises per channel anyway
    imgs /= 255.0
    print(f"   value range [{imgs.min():.3f}, {imgs.max():.3f}]  mean {imgs.mean():.3f}",
          flush=True)

    np.savez_compressed(out_f, images=imgs, coords=s1["coords"], labels=s1["labels"],
                        splits=s1["splits"], names=s1["names"], classes=s1["classes"])
    print(f"\n-> {out_f}  {out_f.stat().st_size/1e6:.0f} MB  images {imgs.shape}", flush=True)

    # check the alignment: file name must start with the class name of its label
    lab = s1["labels"]; cls = [str(c) for c in s1["classes"]]
    bad = [i for i in range(0, len(names), 997)
           if not names[i].startswith(cls[lab[i]])]
    print(f"   alignment spot-check on {len(range(0, len(names), 997))} rows: "
          f"{len(bad)} mismatches between file name and label", flush=True)


if __name__ == "__main__":
    main()
