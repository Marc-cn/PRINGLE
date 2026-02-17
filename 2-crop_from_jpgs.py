#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

CANON_STATES = ["red", "yellow", "green", "red_yellow", "off"]

# -----------------------------
# Helpers
# -----------------------------
def resolve_base(path_str: str) -> str:
    return os.path.normpath(path_str)

def safe_open_image_pil(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")

def open_image_size(path: str) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(path) as im:
            return im.size  # (W, H)
    except Exception:
        return None

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ROBUST bbox clamping (always produces a valid, ≥1×1 crop)
def clamp_box(x, y, w, h, W, H, pad=4) -> Tuple[int,int,int,int]:
    x = float(x); y = float(y); w = float(w); h = float(h)
    if w < 1: w = 1.0
    if h < 1: h = 1.0
    x0 = int(round(x - pad)); y0 = int(round(y - pad))
    x1 = int(round(x + w + pad)); y1 = int(round(y + h + pad))
    x0 = max(0, min(x0, W - 1)); y0 = max(0, min(y0, H - 1))
    x1 = max(0, min(x1, W));     y1 = max(0, min(y1, H))
    if x1 <= x0: x0 = max(0, min(x0, W - 2)); x1 = x0 + 1
    if y1 <= y0: y0 = max(0, min(y0, H - 2)); y1 = y0 + 1
    return (x0, y0, x1, y1)

# -----------------------------
# Variant picking
# -----------------------------
def pick_jpg_matching_variant(img_path: str) -> str:
    """
    Choose the JPG that matches the variant mentioned in img_path.
      .../DE_..._k0.tiff       -> .../DE_..._k0.jpg
      .../DE_..._nativeV2.tif  -> .../DE_..._nativeV2.jpg
      .../DE_....tiff          -> .../DE_....jpg
    If that exact JPG is missing, fall back to:
      if *_k0.*    -> base_k0.jpg, base.jpg, base_nativeV2.jpg
      if *_nativeV2.* -> base_nativeV2.jpg, base.jpg, base_k0.jpg
      else -> base.jpg, base_k0.jpg, base_nativeV2.jpg
    """
    p = Path(img_path)
    folder = p.parent if p.parent.exists() else Path(".")
    name = p.stem  # without extension

    m = re.search(r"_(k0|k1|nativeV2)$", name, re.IGNORECASE)
    suffix = (m.group(1) if m else None)

    def with_suffix(sfx: Optional[str]) -> Path:
        base = name
        if m:
            base = name[:m.start()]  # strip existing suffix
        return folder / (f"{base}_{sfx}.jpg" if sfx else f"{base}.jpg")

    # 1) exact same variant (just .jpg)
    exact = folder / (name + ".jpg")
    if exact.exists():
        return str(exact)

    # 2) fallback order
    if suffix and suffix.lower() in ["k0", "k1"]:
        order = [with_suffix("k0"), with_suffix(None), with_suffix("nativeV2")]
    elif suffix and suffix.lower() == "nativev2":
        order = [with_suffix("nativeV2"), with_suffix(None), with_suffix("k0")]
    else:
        order = [with_suffix(None), with_suffix("k0"), with_suffix("nativeV2")]

    for cand in order:
        if cand.exists():
            return str(cand)

    # last resort
    return str(exact)

def pick_tiff_for_jpg(jpg_path: str) -> Optional[str]:
    """
    Given a JPG path, guess the corresponding TIFF path (same base, variant).
    Returns the first existing candidate (.tiff or .tif) or None.
    """
    p = Path(jpg_path)
    folder = p.parent
    stem = p.stem  # may already include _k0/_nativeV2 etc.

    # Try exact same stem with .tiff / .tif
    for ext in (".tiff", ".tif"):
        cand = folder / f"{stem}{ext}"
        if cand.exists():
            return str(cand)

    # If stem ends with variant, also try the base without the suffix (some trees differ)
    m = re.search(r"_(k0|k1|nativeV2)$", stem, re.IGNORECASE)
    if m:
        base = stem[:m.start()]
        for suff in (m.group(1), None):
            s = f"{base}_{suff}" if suff else base
            for ext in (".tiff", ".tif"):
                cand = folder / f"{s}{ext}"
                if cand.exists():
                    return str(cand)

    # Lastly, try plain base (no suffix) .tiff/.tif
    for ext in (".tiff", ".tif"):
        cand = folder / f"{stem}{ext}"
        if cand.exists():
            return str(cand)

    return None

# -----------------------------
# State normalization
# -----------------------------
def normalize_state(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.lower()
    if s in CANON_STATES: return s
    if s in ["red-yellow", "red_yellow", "redamber", "red_amber", "redandyellow"]:
        return "red_yellow"
    if s in ["dark", "unlit"]:
        return "off"
    return s

# -----------------------------
# Core extraction with scaling
# -----------------------------
def extract_crops_from_json(
    data_root: str,
    json_paths: List[str],
    out_dir: str,
    group_by_state: bool = True,
    pad: int = 4,
    assume_same_resolution: bool = False,
):
    """
    Crops from JPGs using DTLD JSON. If a corresponding TIFF is found, we use its
    size as the "annotation space" and rescale boxes to JPG size. This fixes
    misaligned crops when JPGs were resized during conversion.
    """
    data_root = resolve_base(data_root)
    ensure_dir(out_dir)

    # Cache for sizes to avoid re-opening many times
    jpg_size_cache: Dict[str, Tuple[int,int]] = {}
    tiff_size_cache: Dict[str, Tuple[int,int]] = {}

    for jp in json_paths:
        if not os.path.isfile(jp):
            print(f"[WARN] JSON not found: {jp}")
            continue

        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        print(f"[INFO] {Path(jp).name}: {len(images)} image entries")

        for idx_img, entry in enumerate(images):
            rel = entry.get("image_path") or entry.get("disparity_image_path")
            if not rel:
                continue
            rel = rel.lstrip("./\\")
            raw_path = os.path.join(data_root, rel)

            # Pick JPG that matches the variant referenced by JSON
            jpg_path = pick_jpg_matching_variant(raw_path)
            if not os.path.exists(jpg_path):
                print(f"[WARN] Missing JPG for {rel} -> tried {jpg_path}")
                continue

            # Read JPG size (target crop space)
            if jpg_path in jpg_size_cache:
                W_jpg, H_jpg = jpg_size_cache[jpg_path]
            else:
                sz = open_image_size(jpg_path)
                if not sz:
                    print(f"[WARN] Could not open JPG {jpg_path} to read size.")
                    continue
                W_jpg, H_jpg = sz
                jpg_size_cache[jpg_path] = sz

            # Determine the source (annotation) size
            if assume_same_resolution:
                W_src, H_src = W_jpg, H_jpg
            else:
                # Try to find the matching TIFF next to the JPG
                tiff_path = pick_tiff_for_jpg(jpg_path)
                if tiff_path:
                    if tiff_path in tiff_size_cache:
                        W_src, H_src = tiff_size_cache[tiff_path]
                    else:
                        sz = open_image_size(tiff_path)
                        if not sz:
                            print(f"[WARN] Could not open TIFF {tiff_path} to read size; assuming same as JPG.")
                            W_src, H_src = W_jpg, H_jpg
                        else:
                            W_src, H_src = sz
                            tiff_size_cache[tiff_path] = sz
                else:
                    # No TIFF found — assume the JSON boxes are in JPG space already
                    W_src, H_src = W_jpg, H_jpg

            # Compute scale factors from source (annotation) -> jpg (crop) space
            scale_x = float(W_jpg) / float(max(1, W_src))
            scale_y = float(H_jpg) / float(max(1, H_src))

            # Open JPG for cropping
            try:
                img = safe_open_image_pil(jpg_path)
            except Exception as e:
                print(f"[WARN] Could not open {jpg_path}: {e}")
                continue

            labels = entry.get("labels", [])
            for i, lb in enumerate(labels):
                x, y, w, h = lb.get("x"), lb.get("y"), lb.get("w"), lb.get("h")
                if None in (x, y, w, h):
                    continue
                attrs = lb.get("attributes", {}) or {}
                state = normalize_state(attrs.get("state"))

                # Rescale box from source space to JPG space
                x_s = x * scale_x
                y_s = y * scale_y
                w_s = w * scale_x
                h_s = h * scale_y

                x0, y0, x1, y1 = clamp_box(x_s, y_s, w_s, h_s, W_jpg, H_jpg, pad=pad)
                try:
                    crop = img.crop((x0, y0, x1, y1))
                except Exception:
                    crop = img.crop((max(0, W_jpg-2), max(0, H_jpg-2), max(1, W_jpg-1), max(1, H_jpg-1)))

                dst_dir = os.path.join(out_dir, state) if (group_by_state and state) else out_dir
                ensure_dir(dst_dir)

                json_tag = Path(jp).stem
                state_tag = state if state else "unknown"
                fname = f"{json_tag}__img{idx_img:05d}__box{i:03d}__{state_tag}__{x0}_{y0}_{x1}_{y1}.jpg"
                out_path = os.path.join(dst_dir, fname)

                try:
                    crop.save(out_path, format="JPEG", quality=95)
                except Exception as e:
                    print(f"[WARN] Failed to save crop {out_path}: {e}")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        "Crop traffic-light patches from JPGs using DTLD-style JSON boxes, "
        "rescaling boxes from TIFF size when needed."
    )
    p.add_argument("--data_root", type=str, required=True,
                   help="Folder that contains the image tree referenced by the JSONs.")
    p.add_argument("--json_paths", type=str, nargs="+", required=True,
                   help="DTLD-style JSON file(s).")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Where to save the cropped traffic lights.")
    p.add_argument("--pad", type=int, default=4,
                   help="Padding (in pixels) around each bbox.")
    p.add_argument("--group_by_state", action="store_true",
                   help="Save into subfolders by the signal state.")
    p.add_argument("--assume_same_resolution", action="store_true",
                   help="Force no scaling (use this only if JPGs have the exact same size as the annotated TIFFs).")
    return p.parse_args()

def main():
    args = parse_args()
    extract_crops_from_json(
        data_root=args.data_root,
        json_paths=args.json_paths,
        out_dir=args.out_dir,
        group_by_state=args.group_by_state,
        pad=args.pad,
        assume_same_resolution=args.assume_same_resolution,
    )
    print("[DONE] Crops extracted.")

if __name__ == "__main__":
    main()
