"""
GAMUS preprocessing: raw HDF5 -> tiled patches + manifest CSVs.

Single direction of data flow (raw HDF5 -> processed arrays + manifest lives here,
not in the notebook or the model).

Verified HDF5 layout (earthflow/GAMUS on Hugging Face, cc-by-4.0; checked on real
files DC_01_25 / DC_02_26 / DC_03_26, 2026-09-12):

    images/<split>/<base>_RGB.h5   key 'image',  uint8   (1024, 1024, 3), 0-255
    classes/<split>/<base>_CLS.h5  key 'image',  float32 (1024, 1024), class ids 0..6
    heights/<split>/<base>_AGL.h5  key 'image',  float32 (1024, 1024), nDSM in METRES

The three splits on Hugging Face are named train / val / test (not 'validation').

Each 1024x1024 tile is cut into non-overlapping <patch> x <patch> patches, stored as:
    rgb    -> <out>/<split>/rgb/    PNG uint8     (ImageNet normalization is deferred
                                                   to the training loader)
    height -> <out>/<split>/height/ NPY float32   (nDSM in metres -- already metric,
                                                   no rescaling applied)
    class  -> <out>/<split>/class/  NPY uint8     (land-cover class id)

One manifest CSV per split ties patches together:
    image_path, height_path, class_path, source_dataset, landscape_type, split

landscape_type is a per-tile label used only for later per-bucket evaluation.
Only the two high-confidence class ids drive it (verified consistent across
tiles): id 6 = building, id 3 = tree. Hilly terrain cannot be recovered from an
above-ground nDSM, so tiles are labelled one of {sparse, urban, forested}; the
'hilly' bucket is best served by the coarse-DEM (SRTM) branch at evaluation time.

The official train/val/test split is used as-is: GAMUS built it to be city-aware,
which is the landscape-mixing guarantee the roadmap's "landscape-aware split" ask.

Usage:
    python backend/scripts/preprocess_gamus.py --root gamus_data [--out backend/data/processed/gamus]
"""

import argparse
import csv
import os

import h5py
import numpy as np

from PIL import Image

# --- Verified HDF5 layout ---------------------------------------------------
SPLITS = ["train", "val", "test"]
RGB_SUFFIX = "_RGB.h5"
CLS_SUFFIX = "_CLS.h5"
AGL_SUFFIX = "_AGL.h5"
KEY = "image"  # the single internal key in every .h5 file

# Empirically derived class id -> land-cover name. Only id 6 (building) and id 3
# (tree) are relied on by the landscape heuristic; the rest are recorded for the
# manifest's completeness and re-verifiable with --verbose.
CLASS_NAMES = {
    0: "void",
    1: "ground",
    2: "low-vegetation",
    3: "tree",
    4: "water/other",  # rare (<=0.4% area); label provisional
    5: "road",
    6: "building",
}

BUILDING_ID = 6
TREE_ID = 3


def landscape_type(class_map):
    """Label a tile by object density using the two trusted classes.

    Returns one of sparse / urban / forested. Non-overlapping classes are ignored
    so a wrong side-entry never corrupts the label.
    """
    building_frac = float((class_map == BUILDING_ID).mean())
    tree_frac = float((class_map == TREE_ID).mean())
    object_frac = building_frac + tree_frac
    if object_frac < 0.2:
        return "sparse"
    if building_frac >= tree_frac:
        return "urban"
    return "forested"


def verify_class_map(rgb, class_map, agl, label):
    """Print per-class stats for a tile so the id->name table can be re-checked."""
    print(f"  [{label}] per-class stats (id / frac / mean RGB / mean AGL m):")
    for c in np.unique(class_map):
        m = class_map == c
        rgb_mean = rgb[m].mean(axis=0).round(1)
        print(
            f"    id={int(c):d} ({CLASS_NAMES.get(int(c), '?')}) "
            f"frac={m.mean():.3f} meanRGB={rgb_mean.tolist()} meanAGL={agl[m].mean():.2f}"
        )


def iter_tiles(root, split):
    """Yield (base_name, rgb_path, cls_path, agl_path) for complete triplets."""
    image_dir = os.path.join(root, "images", split)
    class_dir = os.path.join(root, "classes", split)
    height_dir = os.path.join(root, "heights", split)

    rgb_files = sorted(f for f in os.listdir(image_dir) if f.endswith(RGB_SUFFIX))
    for rgb_name in rgb_files:
        base = rgb_name[: -len(RGB_SUFFIX)]
        rgb_path = os.path.join(image_dir, rgb_name)
        cls_path = os.path.join(class_dir, base + CLS_SUFFIX)
        agl_path = os.path.join(height_dir, base + AGL_SUFFIX)
        if not (os.path.exists(cls_path) and os.path.exists(agl_path)):
            print(f"  ! skipping {base}: missing {CLS_SUFFIX}/{AGL_SUFFIX} sibling")
            continue
        yield base, rgb_path, cls_path, agl_path


def process_split(root, out, split, patch, verbose):
    image_dir = os.path.join(root, "images", split)
    if not os.path.isdir(image_dir):
        print(f"  ! skip split '{split}': {image_dir} not found")
        return 0, {}

    out_rgb = os.path.join(out, split, "rgb")
    out_hgt = os.path.join(out, split, "height")
    out_cls = os.path.join(out, split, "class")
    for d in (out_rgb, out_hgt, out_cls):
        os.makedirs(d, exist_ok=True)

    manifest_path = os.path.join(out, f"{split}.csv")
    landscape_counts = {}
    n_tiles = 0
    n_patches = 0

    with open(manifest_path, "w", newline="") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=[
                "image_path",
                "height_path",
                "class_path",
                "source_dataset",
                "landscape_type",
                "split",
            ],
        )
        writer.writeheader()

        for base, rgb_path, cls_path, agl_path in iter_tiles(root, split):
            with h5py.File(rgb_path, "r") as f:
                rgb = f[KEY][()]
            with h5py.File(cls_path, "r") as f:
                cls = f[KEY][()]
            with h5py.File(agl_path, "r") as f:
                agl = f[KEY][()]

            if rgb.shape[:2] != cls.shape != agl.shape:
                print(f"  ! skipping {base}: shape mismatch {rgb.shape} vs {cls.shape} vs {agl.shape}")
                continue
            h, w = rgb.shape[:2]
            if h % patch != 0 or w % patch != 0:
                print(f"  ! skipping {base}: ({h}x{w}) not divisible by patch {patch}")
                continue

            label = landscape_type(cls)
            landscape_counts[label] = landscape_counts.get(label, 0) + 1
            if verbose:
                verify_class_map(rgb, cls, agl, f"{split}/{base}")

            # Cut into (h//patch) x (w//patch) non-overlapping patches.
            for i, y in enumerate(range(0, h, patch)):
                for j, x in enumerate(range(0, w, patch)):
                    rgb_p = rgb[y : y + patch, x : x + patch]
                    agl_p = agl[y : y + patch, x : x + patch]
                    cls_p = cls[y : y + patch, x : x + patch]

                    stem = f"{base}_p{i * (w // patch) + j}"
                    rgb_out = os.path.join(out_rgb, stem + ".png")
                    hgt_out = os.path.join(out_hgt, stem + ".npy")
                    cla_out = os.path.join(out_cls, stem + ".npy")

                    Image.fromarray(rgb_p).save(rgb_out)
                    np.save(hgt_out, agl_p.astype(np.float32))
                    np.save(cla_out, cls_p.astype(np.uint8))

                    writer.writerow(
                        {
                            "image_path": os.path.relpath(rgb_out, out).replace("\\", "/"),
                            "height_path": os.path.relpath(hgt_out, out).replace("\\", "/"),
                            "class_path": os.path.relpath(cla_out, out).replace("\\", "/"),
                            "source_dataset": "GAMUS",
                            "landscape_type": label,
                            "split": split,
                        }
                    )
                    n_patches += 1

            n_tiles += 1
            if n_tiles % 25 == 0:
                print(f"  processed {n_tiles} tiles / {n_patches} patches [{split}]")

    print(f"[{split}] done: {n_tiles} tiles -> {n_patches} patches -> {manifest_path}")
    return n_tiles, landscape_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="gamus_data",
        help="Folder containing images/ classes/ heights/ (default: gamus_data, matching the download notebook)",
    )
    parser.add_argument(
        "--out",
        default="backend/data/processed/gamus",
        help="Where to write patches + manifest CSVs",
    )
    parser.add_argument("--patch", type=int, default=512, help="Patch size (must divide 1024)")
    parser.add_argument("--verbose", action="store_true", help="Print per-class stats per tile")
    args = parser.parse_args()

    if 1024 % args.patch != 0:
        parser.error("--patch must divide 1024 (the native GAMUS tile size)")

    print("=" * 60)
    print("GAMUS preprocessing")
    print(f"root={args.root}  out={args.out}  patch={args.patch}")
    print("=" * 60)

    all_counts = {}
    for split in SPLITS:
        n, counts = process_split(args.root, args.out, split, args.patch, args.verbose)
        all_counts[split] = counts
        # Expect ~12288 = 4x the ~3072 tiles if the full dataset is present.
        print(f"    landscape breakdown: {counts}")

    print("=" * 60)
    print("Done. Manifest CSVs written to", args.out)
    print("Next step: train on the patches via the manifest train.csv "
          "(height is float32 nDSM in metres).")
    if all_counts["train"] and all_counts["val"] and all_counts["test"]:
        print("All three splits processed -- landscape-mixed train/val/test is in place.")


if __name__ == "__main__":
    main()