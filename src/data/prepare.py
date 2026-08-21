"""Turn the raw downloaded datasets into one unified build.

Every dataset ends up in exactly the same format, so that nothing downstream has to
know which hospital an image came from:

    data/processed/<domain>/images/<id>.png    grayscale uint8, SIZE x SIZE
    data/processed/<domain>/masks/<id>.png     0 or 255 only, SIZE x SIZE
    data/processed/manifest.csv                one row per sample

Run:
    python -m src.data.prepare --datasets tn3k ddti
    python -m src.data.prepare --datasets tnscui tg3k     # optional, Step 7

Raw files are only ever read, never modified.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Candidate folder names for each dataset. The first pair that exists is used, so
# small differences between download mirrors do not break the script.
LAYOUTS: dict[str, list[tuple[str, list[str], list[str]]]] = {
    # domain: [(split, image folder candidates, mask folder candidates), ...]
    "tn3k": [
        ("train_pool", ["trainval-image", "trainval_image", "train-image"],
                       ["trainval-mask", "trainval_mask", "train-mask"]),
        ("test",       ["test-image", "test_image"],
                       ["test-mask", "test_mask"]),
    ],
    "ddti": [
        ("test", ["image", "images", "p_image", "DDTI-image"],
                 ["mask", "masks", "p_mask", "DDTI-mask"]),
    ],
    "tnscui": [
        ("test", ["image", "images", "train-image"],
                 ["mask", "masks", "train-mask"]),
    ],
    "tg3k": [
        ("aux", ["thyroid-image", "thyroid_image", "image"],
                ["thyroid-mask", "thyroid_mask", "mask"]),
    ],
}

VAL_FRACTION = 0.10   # held out from TN3K train_pool for all model selection
SPLIT_SEED = 42


# --------------------------------------------------------------------------- io

def find_dir(root: Path, names: list[str]) -> Path | None:
    """Find a sub-directory of `root` whose name matches any of `names`."""
    wanted = {n.lower() for n in names}
    if root.name.lower() in wanted:
        return root
    for path in sorted(root.rglob("*")):
        if path.is_dir() and path.name.lower() in wanted:
            return path
    return None


def list_images(folder: Path) -> dict[str, Path]:
    """Map file stem -> path for every image in `folder` (recursively)."""
    out: dict[str, Path] = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXT:
            out.setdefault(path.stem, path)
    return out


def read_gray(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)          # handles non-ASCII paths
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img


# ------------------------------------------------------------------- resampling

def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if max(img.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Binarise first, then resize by area and re-threshold.

    Several of these datasets ship masks as JPEG, so raw pixel values are not
    exactly 0/255. Thresholding before resizing keeps the nodule area faithful and
    guarantees a strictly binary result.
    """
    binary = (mask > 127).astype(np.float32)
    interp = cv2.INTER_AREA if max(mask.shape[:2]) > size else cv2.INTER_LINEAR
    small = cv2.resize(binary, (size, size), interpolation=interp)
    return ((small >= 0.5).astype(np.uint8)) * 255


# ------------------------------------------------------------ DDTI xml fallback

def rasterise_ddti_xml(xml_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Draw the DDTI contour annotations of one case into a binary mask.

    The DDTI release stores contours inside an <svg> tag as a JSON list of point
    lists, not as an image. This fills those polygons.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return mask
    for svg in root.iter("svg"):
        if not (svg.text or "").strip():
            continue
        try:
            shapes = json.loads(svg.text)
        except json.JSONDecodeError:
            continue
        for shape_dict in shapes:
            pts = shape_dict.get("points", [])
            if len(pts) < 3:
                continue
            poly = np.array([[int(p["x"]), int(p["y"])] for p in pts], dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)
    return mask


def ddti_from_xml(raw_root: Path) -> list[tuple[str, Path, np.ndarray]]:
    """Pair raw DDTI jpgs with rasterised xml annotations."""
    xmls = sorted(raw_root.rglob("*.xml"))
    images = list_images(raw_root)
    out: list[tuple[str, Path, np.ndarray]] = []
    for xml_path in xmls:
        case = re.sub(r"\D", "", xml_path.stem) or xml_path.stem
        members = sorted(s for s in images if re.match(rf"^0*{case}(_\d+)?$", s))
        for stem in members:
            img = read_gray(images[stem])
            if img is None:
                continue
            mask = rasterise_ddti_xml(xml_path, img.shape[:2])
            if mask.max() == 0:
                continue
            out.append((stem, images[stem], mask))
    return out


# ------------------------------------------------------------------- conversion

def convert_dataset(domain: str, size: int, overwrite: bool) -> list[dict]:
    raw_root = RAW / domain
    if not raw_root.exists():
        print(f"[{domain}] skipped: {raw_root} does not exist")
        return []

    img_out = PROCESSED / domain / "images"
    msk_out = PROCESSED / domain / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    found_any = False

    for split, image_names, mask_names in LAYOUTS[domain]:
        image_dir = find_dir(raw_root, image_names)
        mask_dir = find_dir(raw_root, mask_names)

        if image_dir is None or mask_dir is None or image_dir == mask_dir:
            continue
        found_any = True

        images = list_images(image_dir)
        masks = list_images(mask_dir)
        shared = sorted(set(images) & set(masks))
        missing = sorted(set(images) - set(masks))
        if missing:
            print(f"[{domain}/{split}] {len(missing)} images have no mask, dropped "
                  f"(e.g. {missing[:3]})")

        for stem in shared:
            img = read_gray(images[stem])
            mask = read_gray(masks[stem])
            if img is None or mask is None:
                print(f"[{domain}/{split}] unreadable: {stem}")
                continue
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            rows.append(save_sample(domain, split, stem, img, mask, size,
                                    img_out, msk_out, overwrite))

    # DDTI mirrors that ship only raw xml annotations
    if domain == "ddti" and not found_any:
        print("[ddti] no mask folder found, falling back to xml contours")
        for stem, img_path, mask in ddti_from_xml(raw_root):
            img = read_gray(img_path)
            if img is None:
                continue
            rows.append(save_sample(domain, "test", stem, img, mask, size,
                                    img_out, msk_out, overwrite))
        found_any = bool(rows)

    if not found_any:
        print(f"[{domain}] no usable image/mask folders under {raw_root}. "
              f"Expected names like {LAYOUTS[domain][0][1]} and {LAYOUTS[domain][0][2]}.")
        return []

    rows = [r for r in rows if r is not None]
    if domain == "tn3k":
        assign_val_split(rows)

    print(f"[{domain}] {len(rows)} samples written to {PROCESSED / domain}")
    return rows


def save_sample(domain: str, split: str, stem: str, img: np.ndarray, mask: np.ndarray,
                size: int, img_out: Path, msk_out: Path, overwrite: bool) -> dict | None:
    sample_id = f"{domain}_{split}_{stem}"
    img_path = img_out / f"{sample_id}.png"
    msk_path = msk_out / f"{sample_id}.png"

    small_img = resize_image(img, size)
    small_msk = resize_mask(mask, size)

    if overwrite or not img_path.exists():
        cv2.imwrite(str(img_path), small_img)
    if overwrite or not msk_path.exists():
        cv2.imwrite(str(msk_path), small_msk)

    nodule_px = int((small_msk > 0).sum())
    return {
        "domain": domain,
        "split": split,
        "sample_id": sample_id,
        "image": img_path.relative_to(ROOT).as_posix(),
        "mask": msk_path.relative_to(ROOT).as_posix(),
        "src_h": int(img.shape[0]),
        "src_w": int(img.shape[1]),
        "nodule_px": nodule_px,
        "nodule_frac": round(nodule_px / (size * size), 6),
        "empty_mask": int(nodule_px == 0),
    }


def assign_val_split(rows: list[dict]) -> None:
    """Split TN3K train_pool into train / val. This is the only split used for
    model selection, so it is fixed by a seed and never touches a target domain."""
    pool = [r for r in rows if r["split"] == "train_pool"]
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(pool)
    n_val = max(1, int(round(len(pool) * VAL_FRACTION)))
    for r in pool[:n_val]:
        r["split"] = "val"
    for r in pool[n_val:]:
        r["split"] = "train"


# ------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["tn3k", "ddti"],
                    choices=sorted(LAYOUTS))
    ap.add_argument("--size", type=int, default=256, help="output side length")
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite png files that already exist")
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest_path = PROCESSED / "manifest.csv"

    rows: list[dict] = []
    for domain in args.datasets:
        rows += convert_dataset(domain, args.size, args.overwrite)

    if not rows:
        print("\nNothing was converted. Check data/raw/ against the layout in README.md.")
        return

    new = pd.DataFrame(rows)
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        old = old[~old["domain"].isin(args.datasets)]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["domain", "split", "sample_id"]).reset_index(drop=True)
    new.to_csv(manifest_path, index=False)

    print(f"\nmanifest: {manifest_path}  ({len(new)} rows)")
    print(new.groupby(["domain", "split"]).size().to_string())
    empties = int(new["empty_mask"].sum())
    if empties:
        print(f"\nwarning: {empties} samples have an empty mask after resizing. "
              f"verify_data.py lists them.")


if __name__ == "__main__":
    main()
