"""Shared DTLD crop/filter, demosaic/crop builders, laser paste, attack combos for clean/."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import yaml
except ImportError:
    yaml = None

CANON_STATES = ["red", "yellow", "green", "red_yellow", "off"]


# Default paths.
#
# Layout (repo root):
#
#   data/
#     dtld/                 # place raw DTLD frames here
#     labels/               # place DTLD_Labels_v2.0/v2.0 JSON here
#     laser/                # place Red/ Green/ IR_modulated/ traces here
#     crops_jpg/            # written by scripts/process_data.py
#     synth_jpg/            # optional; written by scripts/process_data.py
#   output/                 # eval outputs (laser ASR, consistency)
#   checkpoints/            # trained .pt checkpoints
#   scripts/                # Python entrypoints + helpers
#

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PACKAGE_ROOT / "data"
OUTPUT_ROOT = PACKAGE_ROOT / "output"

DEFAULT_DTLD_ROOT = str(DATA_ROOT / "dtld")
DEFAULT_LABELS_DIR = str(DATA_ROOT / "labels")
DEFAULT_LASER_TRACES = str(DATA_ROOT / "laser")
DEFAULT_CROPS_DIR = str(DATA_ROOT / "crops_jpg")
DEFAULT_SYNTH_CACHE = str(DATA_ROOT / "synth_jpg")
# process_data --out_root writes <out_root>/crops_jpg and <out_root>/synth_jpg
DEFAULT_PROCESSED_ROOT = str(DATA_ROOT)

DEFAULT_TRAIN_ROOT = str(PACKAGE_ROOT / "checkpoints")
DEFAULT_LASER_OUT = str(OUTPUT_ROOT / "laser_injection")
DEFAULT_CONS_OUT = str(OUTPUT_ROOT / "cardinality_consistency")
# Package root doubles as the "code root" for relative eval outputs.
DEFAULT_CODE_ROOT = str(PACKAGE_ROOT)
# Optional: override with CONDA_ENV env var; name-only activate also works.
DEFAULT_CONDA_ENV = os.environ.get("CONDA_ENV", "traffic-vlm")

DEFAULT_ATTR_FILTERS = {
    "aspects": "three_aspects",
    "direction": "front",
    "orientation": "vertical",
    "pictogram": "circle",
    "occlusion": "not_occluded",
}

DTLD_SPLITS = ("train", "test", "all")
DTLD_CITIES = (
    "Berlin", "Bochum", "Bremen", "Dortmund", "Duesseldorf",
    "Essen", "Frankfurt", "Fulda", "Hannover", "Kassel", "Koeln",
)
DTLD_LABEL_CANDIDATES = {
    "all": [
        "DTLD_Labels_v2.0/v2.0/DTLD_all.json",
        "DTLD_Labels_v2.0/DTLD_all.json",
        "DTLD_Labels/DTLD_all.yml",
    ],
    "train": [
        "DTLD_Labels_v2.0/v2.0/DTLD_train.json",
        "DTLD_Labels_v2.0/DTLD_train.json",
        "DTLD_Labels/DTLD_train.yml",
    ],
    "test": [
        "DTLD_Labels_v2.0/v2.0/DTLD_test.json",
        "DTLD_Labels_v2.0/DTLD_test.json",
        "DTLD_Labels/DTLD_test.yml",
    ],
}
DTLD_SUFFIX_TO_STATE = {
    0: "off", 2: "off", 8: "red", 10: "red_yellow", 12: "red_yellow",
    18: "red_yellow", 20: "yellow", 22: "yellow", 28: "yellow",
    40: "green", 42: "green", 48: "green",
}

# Vertical EU three-aspect module slots (top → bottom).
VERTICAL_THREE_ASPECT_SLOTS = {"red": 0, "yellow": 1, "green": 2}
VERTICAL_THREE_ASPECT_OPPOSITE = {0: 2, 1: 1, 2: 0}
MODULE_NAME_TO_SLOT = {"top": 0, "middle": 1, "bottom": 2}
SLOT_TO_MODULE_NAME = {0: "top", 1: "middle", 2: "bottom"}
# Which modules are illuminated for each canonical state.
STATE_ON_MODULES = {
    "red": ("top",),
    "yellow": ("middle",),
    "green": ("bottom",),
    "red_yellow": ("top", "middle"),
    "off": (),
}
# Laser colors injectable into off modules only.
INJECTABLE_LASER_COLORS = ("red", "green", "ir")
ALL_MODULES = ("top", "middle", "bottom")

CROP_FILENAME_RE = re.compile(
    r"^(?P<city>.+?)__img(?P<img_idx>\d+)__box(?P<box_idx>\d+)__"
    r"(?P<folder_state>.+?)__(?P<x0>\d+)_(?P<y0>\d+)_(?P<x1>\d+)_(?P<y1>\d+)\.jpg$",
    re.IGNORECASE,
)

# normalizes a state string to a canonical form
def normalize_state(state: str | None) -> str | None:
    if not state:
        return None
    s = state.strip().lower().replace("-", "_")
    if s in ("unknown", "unlabeled"):
        return None
    if s in CANON_STATES:
        return s
    if s in ("red-yellow", "redamber", "red_amber", "redandyellow"):
        return "red_yellow"
    if s in ("dark", "unlit"):
        return "off"
    return None

# normalizes a relative path string
def normalize_rel_path(path_str: str) -> str:
    rel = path_str.strip().replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    return rel

# checks if attributes match required filters
def attributes_match(attrs: dict, required: dict[str, str]) -> bool:
    if not required:
        return True
    for key, expected in required.items():
        if attrs.get(key) != expected:
            return False
    return True

# checks if attributes match vertical three-aspect placement
def uses_vertical_three_aspect_placement(attrs: dict | None) -> bool:
    if not attrs:
        return False
    return (
        attrs.get("aspects") == "three_aspects"
        and attrs.get("orientation") == "vertical"
    )

# returns modules that are off (valid laser-injection targets) for a state
def off_modules_for_state(state: str | None) -> tuple[str, ...]:
    """Return modules that are off (valid laser-injection targets) for a state."""
    canon = normalize_state(state)
    if canon is None or canon not in STATE_ON_MODULES:
        return ()
    on = set(STATE_ON_MODULES[canon])
    return tuple(m for m in ALL_MODULES if m not in on)

# returns valid (laser_color, target_module) pairs for injecting into off modules only
def valid_laser_injections(
    state: str | None,
    laser_colors: tuple[str, ...] = INJECTABLE_LASER_COLORS,
) -> list[tuple[str, str]]:
    """
    Valid (laser_color, target_module) pairs for injecting into off modules only.
    """
    return [
        (laser, module)
        for module in off_modules_for_state(state)
        for laser in laser_colors
    ]

# loads attribute filters from a JSON file
def load_attr_filters(path: str | None) -> dict[str, str]:
    if not path:
        return dict(DEFAULT_ATTR_FILTERS)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Attribute filter JSON must be an object, got {type(data).__name__}")
    return {str(k): str(v) for k, v in data.items()}

# stretches 16-bit images to 8-bit
def stretch_16bit_to_8bit(img16, low_p=2, high_p=98, gamma=0.9):
    img = img16.astype(np.float32)
    p_lo, p_hi = np.percentile(img, (low_p, high_p))
    if p_hi <= p_lo:
        p_lo, p_hi = float(img.min()), float(img.max())
    img = np.clip((img - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    if gamma != 1.0:
        img = np.power(img, gamma)
    return (img * 255.0 + 0.5).astype(np.uint8)

# loads a DTLD .tiff as a color (demosaiced) RGB image
def load_dtld_color(path: str, use_stretch: bool = True) -> np.ndarray:
    """Load a DTLD *_k0.tiff Bayer frame as an 8-bit RGB image."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"Could not read DTLD TIFF: {path}")
    if img.ndim != 2:
        raise ValueError(
            f"Expected raw Bayer TIFF (H×W), got shape {img.shape} for {path}"
        )
    bgr16 = cv2.cvtColor(img, cv2.COLOR_BayerGB2BGR)
    if use_stretch:
        channels = [
            stretch_16bit_to_8bit(bgr16[:, :, c], low_p=2, high_p=98, gamma=0.9)
            for c in range(3)
        ]
        bgr8 = cv2.merge(channels)
    else:
        bgr8 = np.right_shift(bgr16, 4).astype(np.uint8)
    return cv2.cvtColor(bgr8, cv2.COLOR_BGR2RGB)

# loads a DTLD .tiff as a BGR image
def load_dtld_bgr(path: str, use_stretch: bool = True) -> np.ndarray:
    rgb = load_dtld_color(path, use_stretch=use_stretch)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# resolves a DTLD image path from a raw path and dataset root
def resolve_dtld_image_path(raw_path: str, dtld_root: str) -> str | None:
    raw = raw_path.strip().replace("\\", "/")
    candidates = []
    rel = normalize_rel_path(raw)
    if rel:
        candidates.append(os.path.join(dtld_root, rel))
    for city in DTLD_CITIES:
        marker = f"/{city}/"
        idx = raw.find(marker)
        if idx >= 0:
            candidates.append(os.path.join(dtld_root, raw[idx + 1:]))
    for marker in ("/DTLD_final/", "/DTLD/"):
        idx = raw.find(marker)
        if idx >= 0:
            candidates.append(os.path.join(dtld_root, raw[idx + len(marker):]))
    basename = Path(raw).name
    if basename:
        candidates.append(os.path.join(dtld_root, basename))
    seen = set()
    for cand in candidates:
        cand = os.path.normpath(cand)
        if cand in seen:
            continue
        seen.add(cand)
        if Path(cand).is_file():
            return str(Path(cand).resolve())
    return None

# finds a DTLD label file for a split
def find_dtld_label_file(dtld_root: str, split: str) -> str:
    for rel in DTLD_LABEL_CANDIDATES.get(split, []):
        path = os.path.join(dtld_root, rel)
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        f"No DTLD label file found for split={split!r} under {dtld_root}. "
        f"Tried: {DTLD_LABEL_CANDIDATES.get(split, [])}. "
        "Pass an explicit labels file path."
    )

# maps a DTLD class ID to a canonical state 
def dtld_class_id_to_state(class_id: int) -> str | None:
    suffix = int(class_id) % 100
    if suffix in DTLD_SUFFIX_TO_STATE:
        return DTLD_SUFFIX_TO_STATE[suffix]
    digits = str(int(class_id)).zfill(6)
    return {
        "0": "off", "1": "red_yellow", "2": "red",
        "3": "yellow", "4": "green",
    }.get(digits[4])


# clamps a bounding box to the image dimensions
def clamp_bbox(bbox, img_h, img_w):
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(int(img_w), int(x2))
    y2 = min(int(img_h), int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

# expands a bounding box with a padding
def expand_bbox_with_pad(bbox, img_w, img_h, pad=4):
    x1, y1, x2, y2 = bbox
    return clamp_bbox((x1 - pad, y1 - pad, x2 + pad, y2 + pad), img_h, img_w)

# checks if a label passes the filters
def label_passes_filters(
    attrs: dict,
    canon_state: str | None,
    eval_classes: tuple[str, ...],
    required_attrs: dict[str, str] | None,
    exclude_unknown: bool,
) -> bool:
    if exclude_unknown and canon_state is None:
        return False
    if canon_state not in eval_classes:
        return False
    if required_attrs and not attributes_match(attrs, required_attrs):
        return False
    return True

# appends a frame record to the list of records
def _append_frame_record(
    records: list,
    seen_records: set,
    img_path: str,
    canon: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    attrs: dict,
    extra: dict | None = None,
):
    if x2 <= x1 or y2 <= y1:
        return
    key = (img_path, canon, x1, y1, x2, y2)
    if key in seen_records:
        return
    seen_records.add(key)
    rec = {
        "image_path": img_path,
        "label": canon,
        "bbox": (x1, y1, x2, y2),
        "attributes": dict(attrs),
    }
    if extra:
        rec.update(extra)
    records.append(rec)

# loads DTLD JSON annotations
def load_dtld_json_annotations(
    json_path: str,
    dtld_root: str,
    eval_classes: tuple[str, ...],
    required_attrs: dict[str, str] | None = None,
    exclude_unknown: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images", []) if isinstance(data, dict) else data
    records = []
    seen_records = set()
    stats: dict[str, int] = defaultdict(int)

    for entry in images:
        raw_path = entry.get("image_path") or ""
        if not raw_path:
            continue
        if "_nativev2" in raw_path.lower() and "_k0" not in raw_path.lower():
            continue
        img_path = resolve_dtld_image_path(raw_path, dtld_root)
        if not img_path:
            stats["missing_image"] += 1
            continue

        for lb in entry.get("labels", []) or []:
            stats["labels_seen"] += 1
            attrs = lb.get("attributes", {}) or {}
            canon = normalize_state(attrs.get("state"))
            if not label_passes_filters(attrs, canon, eval_classes, required_attrs, exclude_unknown):
                if exclude_unknown and canon is None:
                    stats["skipped_unknown"] += 1
                elif canon not in eval_classes:
                    stats["skipped_class"] += 1
                elif required_attrs and not attributes_match(attrs, required_attrs):
                    stats["skipped_attrs"] += 1
                continue
            try:
                x = float(lb["x"])
                y = float(lb["y"])
                w = float(lb["w"])
                h = float(lb["h"])
            except (KeyError, TypeError, ValueError):
                stats["skipped_bbox"] += 1
                continue
            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + w))
            y2 = int(round(y + h))
            _append_frame_record(
                records, seen_records, img_path, canon, x1, y1, x2, y2, attrs,
                extra={
                    "track_id": lb.get("track_id"),
                    "unique_id": lb.get("unique_id"),
                },
            )
            stats["accepted"] += 1

    return records, dict(stats)

# loads DTLD YAML annotations
def load_dtld_yaml_annotations(
    yaml_path: str,
    dtld_root: str,
    eval_classes: tuple[str, ...],
    required_attrs: dict[str, str] | None = None,
    exclude_unknown: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    if yaml is None:
        raise ImportError("PyYAML is required to read DTLD YAML label files.")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {yaml_path}, got {type(data).__name__}")

    records = []
    seen_records = set()
    stats: dict[str, int] = defaultdict(int)

    for entry in data:
        raw_path = entry.get("path") or ""
        if not raw_path:
            continue
        img_path = resolve_dtld_image_path(raw_path, dtld_root)
        if not img_path:
            stats["missing_image"] += 1
            continue

        for obj in entry.get("objects", []) or []:
            stats["labels_seen"] += 1
            try:
                class_id = int(obj["class_id"])
                x = float(obj["x"])
                y = float(obj["y"])
                w = float(obj["width"])
                h = float(obj["height"])
            except (KeyError, TypeError, ValueError):
                stats["skipped_bbox"] += 1
                continue

            canon = dtld_class_id_to_state(class_id)
            attrs = {"state": canon or "unknown"}
            if not label_passes_filters(attrs, canon, eval_classes, required_attrs, exclude_unknown):
                if exclude_unknown and canon is None:
                    stats["skipped_unknown"] += 1
                elif canon not in eval_classes:
                    stats["skipped_class"] += 1
                elif required_attrs and not attributes_match(attrs, required_attrs):
                    stats["skipped_attrs"] += 1
                continue

            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + w))
            y2 = int(round(y + h))
            _append_frame_record(records, seen_records, img_path, canon, x1, y1, x2, y2, attrs)
            stats["accepted"] += 1

    return records, dict(stats)

# loads DTLD annotations from a JSON or YAML file
def load_dtld_annotations(
    dtld_root: str,
    eval_classes: tuple[str, ...],
    split: str = "test",
    labels_file: str | None = None,
    required_attrs: dict[str, str] | None = None,
    exclude_unknown: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    label_path = labels_file or find_dtld_label_file(dtld_root, split)
    if label_path.lower().endswith(".json"):
        return load_dtld_json_annotations(
            label_path, dtld_root, eval_classes, required_attrs, exclude_unknown,
        )
    if label_path.lower().endswith((".yml", ".yaml")):
        return load_dtld_yaml_annotations(
            label_path, dtld_root, eval_classes, required_attrs, exclude_unknown,
        )
    raise ValueError(f"Unsupported DTLD label format: {label_path}")

# loads DTLD benign crop items from a dataset root
def load_dtld_benign_crop_items(
    dtld_root: str,
    eval_classes: tuple[str, ...],
    crops_dir,
    split: str = "test",
    labels_file: str | None = None,
    max_per_class: int | None = None,
    pad: int = 4,
    required_attrs: dict[str, str] | None = None,
    exclude_unknown: bool = True,
    use_stretch: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    """Demosaic DTLD TIFF frames, crop filtered labels, write benign crop JPGs.
    """
    if required_attrs is None:
        required_attrs = DEFAULT_ATTR_FILTERS

    records, load_stats = load_dtld_annotations(
        dtld_root=dtld_root,
        eval_classes=eval_classes,
        split=split,
        labels_file=labels_file,
        required_attrs=required_attrs,
        exclude_unknown=exclude_unknown,
    )

    by_class = defaultdict(list)
    for rec in records:
        by_class[rec["label"]].append(rec)

    # Apply per-class caps first, then group by frame so each TIFF is loaded once.
    selected: list[tuple[str, dict]] = []
    for benign_color in eval_classes:
        pool = sorted(by_class[benign_color], key=lambda r: r["image_path"])
        if max_per_class is not None:
            pool = pool[:max_per_class]
        print(
            f"{benign_color} benign (DTLD): {len(pool)} crop(s) "
            f"(from {len(by_class[benign_color])} filtered annotation(s))"
        )
        for rec in pool:
            selected.append((benign_color, rec))

    by_frame: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for benign_color, rec in selected:
        by_frame[rec["image_path"]].append((benign_color, rec))

    items = []
    crop_stats: dict[str, int] = defaultdict(int)
    crop_stats.update(load_stats)
    crop_stats["unique_frames"] = len(by_frame)

    os.makedirs(crops_dir, exist_ok=True)
    for benign_color in eval_classes:
        os.makedirs(os.path.join(crops_dir, benign_color), exist_ok=True)

    for frame_idx, (frame_path, group) in enumerate(sorted(by_frame.items()), start=1):
        try:
            bgr = load_dtld_bgr(frame_path, use_stretch=use_stretch)
        except Exception as exc:
            crop_stats["frame_load_failed"] += 1
            print(f"  skip frame {frame_path}: {exc}")
            continue

        img_h, img_w = bgr.shape[:2]
        for benign_color, rec in group:
            bbox = expand_bbox_with_pad(rec["bbox"], img_w, img_h, pad=pad)
            if bbox is None:
                crop_stats["skipped_empty_crop"] += 1
                continue
            x1, y1, x2, y2 = bbox
            crop_bgr = bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                crop_stats["skipped_empty_crop"] += 1
                continue

            stem = f"{Path(frame_path).stem}_x{x1}_y{y1}_x{x2}_y{y2}"
            fname = f"{stem}.jpg"
            out_path = os.path.join(crops_dir, benign_color, fname)
            cv2.imwrite(out_path, crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            items.append({
                "image": fname,
                "path": out_path,
                "benign_color": benign_color,
                "gt_state": benign_color,
                "source": "dtld",
                "benign_dir": dtld_root,
                "frame_path": frame_path,
                "bbox": [x1, y1, x2, y2],
                "attributes": rec.get("attributes", {}),
            })
            crop_stats["crops_written"] += 1

        # Free the demosaiced frame before loading the next one.
        del bgr
        if frame_idx % 25 == 0:
            import gc
            gc.collect()
        if frame_idx % 100 == 0:
            print(
                f"  cropped {frame_idx}/{len(by_frame)} frames "
                f"({crop_stats['crops_written']} crops)..."
            )

    if not items:
        raise FileNotFoundError(
            f"No DTLD benign crops produced under {dtld_root} (split={split}). "
            f"Stats: {dict(crop_stats)}"
        )
    return items, dict(crop_stats)


# DTLDLabelIndex class - lazy loader for per-city DTLD JSON label files
class DTLDLabelIndex:
    """loader for per-city DTLD JSON label files."""

    def __init__(self, labels_dir: str | Path):
        self.labels_dir = Path(labels_dir)
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"DTLD labels directory not found: {self.labels_dir}")
        self._city_cache: dict[str, dict] = {}

    def _load_city(self, city_tag: str) -> dict | None:
        if city_tag in self._city_cache:
            return self._city_cache[city_tag]
        json_path = self.labels_dir / f"{city_tag}.json"
        if not json_path.is_file():
            self._city_cache[city_tag] = None
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._city_cache[city_tag] = data
        return data

    def lookup_label(self, filename: str) -> dict[str, Any] | None:
        m = CROP_FILENAME_RE.match(Path(filename).name)
        if not m:
            return None
        parsed = {
            "city_tag": m["city"],
            "image_idx": int(m["img_idx"]),
            "box_idx": int(m["box_idx"]),
            "folder_state": m["folder_state"].lower().replace("-", "_"),
            "bbox_xyxy_padded": [int(m[k]) for k in ("x0", "y0", "x1", "y1")],
        }
        city_data = self._load_city(parsed["city_tag"])
        if not city_data:
            return None
        images = city_data.get("images", [])
        if parsed["image_idx"] >= len(images):
            return None
        frame = images[parsed["image_idx"]]
        labels = frame.get("labels", [])
        if parsed["box_idx"] >= len(labels):
            return None
        label = labels[parsed["box_idx"]]
        attrs = label.get("attributes", {}) or {}
        return {
            **parsed,
            "image_path": frame.get("image_path"),
            "attributes": attrs,
            "track_id": label.get("track_id"),
            "unique_id": label.get("unique_id"),
            "bbox_xywh": [
                float(label.get("x", 0)),
                float(label.get("y", 0)),
                float(label.get("w", 0)),
                float(label.get("h", 0)),
            ],
        }

# parses a crop filename from the dtld dataset into a dictionary of metadata
def parse_crop_filename(filename: str) -> dict[str, Any] | None:
    m = CROP_FILENAME_RE.match(Path(filename).name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "city_tag": d["city"],
        "image_idx": int(d["img_idx"]),
        "box_idx": int(d["box_idx"]),
        "folder_state": d["folder_state"].lower().replace("-", "_"),
        "bbox_xyxy_padded": [int(d[k]) for k in ("x0", "y0", "x1", "y1")],
    }

# scans crops_jpg and filters via per-city JSON lookup for training scripts
def scan_filtered_crop_records(
    image_dir: str | Path,
    labels_dir: str | Path,
    required_attrs: dict[str, str] | None = None,
    exclude_unknown: bool = True,
    use_json_state: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Scan crops_jpg and filter via per-city JSON lookup."""
    if required_attrs is None:
        required_attrs = DEFAULT_ATTR_FILTERS

    image_base = Path(image_dir)
    if not image_base.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_base}")

    index = DTLDLabelIndex(labels_dir)
    records: list[dict[str, Any]] = []
    stats: dict[str, int] = defaultdict(int)

    for state_dir in sorted(image_base.iterdir()):
        if not state_dir.is_dir():
            continue
        folder_state = state_dir.name.lower()
        stats["scanned_dirs"] += 1

        if exclude_unknown and folder_state == "unknown":
            for _ in state_dir.glob("*.jpg"):
                stats["skipped_unknown_folder"] += 1
            continue
        if folder_state not in CANON_STATES:
            for _ in state_dir.glob("*.jpg"):
                stats["skipped_non_canon_folder"] += 1
            continue

        for img_path in sorted(state_dir.glob("*.jpg")):
            stats["scanned_images"] += 1
            rel = str(img_path.relative_to(image_base))
            meta = index.lookup_label(img_path.name)
            if meta is None:
                stats["lookup_failed"] += 1
                continue

            attrs = meta["attributes"]
            json_state = normalize_state(attrs.get("state"))
            if exclude_unknown and json_state is None:
                stats["skipped_unknown_state"] += 1
                continue
            if json_state not in CANON_STATES:
                stats["skipped_bad_state"] += 1
                continue
            if not attributes_match(attrs, required_attrs):
                stats["skipped_attrs"] += 1
                continue

            train_state = json_state if use_json_state else folder_state
            records.append({
                "filename": rel,
                "state": train_state,
                "folder_state": folder_state,
                "city_tag": meta["city_tag"],
                "image_idx": meta["image_idx"],
                "box_idx": meta["box_idx"],
                "attributes": attrs,
                "track_id": meta.get("track_id"),
                "unique_id": meta.get("unique_id"),
                "frame_image_path": meta.get("image_path"),
                "bbox_xywh": meta.get("bbox_xywh"),
                "bbox_xyxy_padded": meta.get("bbox_xyxy_padded"),
            })
            stats["accepted"] += 1

    return records, dict(stats)


# paste_laser_pattern - attribute-aware laser pattern placement

# calculates the y-range for a slot in a cropped image
def _slot_y_range(crop_h: int, slot: int, n_slots: int = 3, margin_frac: float = 0.08):
    slot_h = crop_h / n_slots
    y1 = int(round(slot * slot_h + margin_frac * slot_h))
    y2 = int(round((slot + 1) * slot_h - margin_frac * slot_h))
    y1 = max(0, min(y1, crop_h - 2))
    y2 = max(y1 + 1, min(y2, crop_h))
    return y1, y2

# pastes a laser pattern onto a traffic-light crop using HSV color thresholding
def _paste_laser_hsv(sim_image_bgr, pattern_bgr, flip_y, sim_threshold):
    hsv_image = cv2.cvtColor(sim_image_bgr, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, sim_threshold])
    upper_bound = np.array([179, 255, 255])
    result_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(contours[0])
    x1, y1, x2, y2 = x, y, x + w, y + h
    attacked = sim_image_bgr.copy()
    resized = cv2.resize(pattern_bgr, (x2 - x1, y2 - y1))
    if flip_y:
        h_img = attacked.shape[0]
        attacked[h_img - y2:h_img - y1, x1:x2] = resized
    else:
        attacked[y1:y2, x1:x2] = resized
    return attacked


# resolves the target slot to paste the laser pattern into
def _resolve_target_slot(
    benign_state: str | None,
    flip_y: bool,
    target_module: str | None,
) -> int | None:
    """Resolve paste slot from explicit module name or lit/opposite fallback."""
    if target_module is not None:
        key = str(target_module).strip().lower()
        if key not in MODULE_NAME_TO_SLOT:
            raise ValueError(
                f"Unknown target_module={target_module!r}; "
                f"expected one of {tuple(MODULE_NAME_TO_SLOT)}"
            )
        return MODULE_NAME_TO_SLOT[key]

    if not benign_state:
        return None
    lit_slot = VERTICAL_THREE_ASPECT_SLOTS.get(benign_state)
    if lit_slot is None:
        # Fully off (or multi-lit states without a single lit slot): no default.
        return None
    return VERTICAL_THREE_ASPECT_OPPOSITE[lit_slot] if flip_y else lit_slot

# crops a masked laser pattern to just the active pixels
def _crop_laser_content(pattern_bgr, min_value: int = 8):
    """Crop a masked laser pattern to its non-black content bbox + binary mask."""
    if pattern_bgr is None or pattern_bgr.size == 0:
        return None, None
    content = pattern_bgr.max(axis=2)
    ys, xs = np.where(content > min_value)
    if len(ys) == 0:
        return None, None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = pattern_bgr[y0:y1, x0:x1]
    mask = (crop.max(axis=2) > min_value).astype(np.uint8) * 255
    return crop, mask

# pastes a laser pattern onto a traffic-light crop using a target slot
def _paste_laser_slot(
    sim_image_bgr,
    pattern_bgr,
    benign_state: str | None = None,
    flip_y: bool = False,
    target_module: str | None = None,
):
    h, w = sim_image_bgr.shape[:2]
    target_slot = _resolve_target_slot(benign_state, flip_y, target_module)
    if target_slot is None:
        return None
    y1, y2 = _slot_y_range(h, target_slot)
    x_margin = int(round(w * 0.22))
    x1 = max(0, x_margin)
    x2 = min(w, w - x_margin)
    if x2 <= x1 or y2 <= y1:
        return None

    content, content_mask = _crop_laser_content(pattern_bgr)
    if content is None:
        return None

    slot_w, slot_h = x2 - x1, y2 - y1
    resized = cv2.resize(content, (slot_w, slot_h), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(content_mask, (slot_w, slot_h), interpolation=cv2.INTER_NEAREST)
    attacked = sim_image_bgr.copy()
    region = attacked[y1:y2, x1:x2]
    # Only overwrite lit laser pixels — avoid filling the module with black canvas.
    keep = mask_r > 0
    region[keep] = resized[keep]
    attacked[y1:y2, x1:x2] = region
    return attacked

# pastes a laser pattern onto a traffic-light crop using attribute-aware placement
def paste_laser_pattern(
    sim_image_bgr,
    pattern_bgr,
    flip_y: bool = False,
    sim_threshold: int = 100,
    attributes: dict | None = None,
    benign_state: str | None = None,
    target_module: str | None = None,
):
    """
    Paste laser pattern onto a traffic-light crop.
    """
    # Explicit module targeting works for both DTLD and sim crops.
    if target_module is not None:
        attacked = _paste_laser_slot(
            sim_image_bgr,
            pattern_bgr,
            benign_state=benign_state,
            flip_y=False,
            target_module=target_module,
        )
        return attacked

    if attributes and uses_vertical_three_aspect_placement(attributes):
        attacked = _paste_laser_slot(
            sim_image_bgr,
            pattern_bgr,
            benign_state=benign_state,
            flip_y=flip_y,
            target_module=None,
        )
        if attacked is not None:
            return attacked
    return _paste_laser_hsv(sim_image_bgr, pattern_bgr, flip_y, sim_threshold)


# crop / demosaic builders

JPEG_QUALITY = 95


def demosaic_gbrg_tiff_to_rgb8(tiff_path: str) -> Optional[np.ndarray]:
    """Demosaic a raw Bayer (GBRG) TIFF to 8-bit RGB (alias of load_dtld_color)."""
    try:
        return load_dtld_color(tiff_path, use_stretch=True)
    except (OSError, ValueError):
        return None


def convert_tiff_to_frame(tiff_path: str, use_color: bool) -> Optional[np.ndarray]:
    """Load a k0/k1 TIFF as demosaiced RGB (color path used by process_data)."""
    if use_color:
        return demosaic_gbrg_tiff_to_rgb8(tiff_path)
    # Grayscale fallback for API compatibility (process_data uses color only).
    img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] >= 3 else img[:, :, 0]
    else:
        gray = img
    if gray.dtype != np.uint8:
        gray = stretch_16bit_to_8bit(gray, low_p=2, high_p=98, gamma=0.9)
    return gray

# resolves the base path of a path string
def resolve_base(path_str: str) -> str:
    return os.path.normpath(path_str)

# ensures a directory exists
def ensure_dir(p: str | Path) -> None:
    os.makedirs(p, exist_ok=True)

# clamps a bounding box to the image dimensions
def clamp_box(x, y, w, h, W, H, pad=4) -> Tuple[int, int, int, int]:
    """xywh -> padded xyxy via expand_bbox_with_pad."""
    w = max(float(w), 1.0)
    h = max(float(h), 1.0)
    bbox = expand_bbox_with_pad((float(x), float(y), float(x) + w, float(y) + h), W, H, pad=pad)
    if bbox is not None:
        return bbox
    x0 = max(0, min(int(round(float(x))), max(W - 2, 0)))
    y0 = max(0, min(int(round(float(y))), max(H - 2, 0)))
    return x0, y0, min(x0 + 1, W), min(y0 + 1, H)

# resolves a TIFF path from a data root and relative image path
def resolve_tiff_path(data_root: str, rel_image_path: str) -> str:
    """Join dataset root + relative TIFF path (no existence check)."""
    return os.path.join(resolve_base(data_root), rel_image_path)

# crops a frame from an image
def crop_frame(
    frame: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    W: int,
    H: int,
    pad: int,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = clamp_box(x, y, w, h, W, H, pad=pad)
    crop = frame[y0:y1, x0:x1]
    if crop.ndim == 2:
        return Image.fromarray(crop).convert("RGB"), (x0, y0, x1, y1)
    return Image.fromarray(crop), (x0, y0, x1, y1)

# discovers per-city DTLD JSON files
def discover_city_jsons(data_root: str, labels_dir: Optional[str] = None) -> List[str]:
    """Find per-city DTLD JSON files (Berlin.json, Bochum.json, ...)."""
    root = Path(labels_dir) if labels_dir else Path(data_root)
    skip = {
        "DTLD_train", "DTLD_test", "DTLD_all", "grid",
        "DTLD_train.json", "DTLD_test.json", "DTLD_all.json", "grid.json",
    }
    found = []
    for p in sorted(root.glob("*.json")):
        if p.stem in skip:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "images" in data:
            found.append(str(p.resolve()))
    return found


# eval attack combos

DEFAULT_EVAL_ATTACK_COMBOS: tuple[str, ...] = (
    "red_laser_on_green_top",
    "ir_laser_on_green_top",
    "ir_laser_on_green_middle",
    "green_laser_on_red_bottom",
    "ir_laser_on_red_middle",
    "ir_laser_on_red_bottom",
    "red_laser_on_yellow_top",
    "green_laser_on_yellow_bottom",
    "ir_laser_on_yellow_top",
    "ir_laser_on_yellow_bottom",
    "red_laser_on_off_top",
    "green_laser_on_off_bottom",
    "ir_laser_on_off_top",
    "ir_laser_on_off_middle",
    "ir_laser_on_off_bottom",
    "green_laser_on_red_yellow_bottom",
    "ir_laser_on_red_yellow_bottom",
)

# parses a combo key into a tuple of laser, benign, and module
def parse_combo_key(combo_key: str) -> tuple[str, str, str] | None:
    """Parse ``{laser}_laser_on_{benign}_{module}`` → (laser, benign, module)."""
    parts = str(combo_key).split("_laser_on_")
    if len(parts) != 2:
        return None
    laser = parts[0]
    rest = parts[1]
    for module in ("top", "middle", "bottom"):
        suffix = f"_{module}"
        if rest.endswith(suffix):
            benign = rest[: -len(suffix)]
            return laser, benign, module
    return None

# maps benign_state → [(laser_color, target_module), ...] from combo keys   
def eval_injections_by_state(
    combos: tuple[str, ...] | list[str] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Map benign_state → [(laser_color, target_module), ...] from combo keys."""
    if combos is None:
        combos = DEFAULT_EVAL_ATTACK_COMBOS
    by_state: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for key in combos:
        parsed = parse_combo_key(key)
        if parsed is None:
            continue
        laser, benign, module = parsed
        trip = (benign, laser, module)
        if trip in seen:
            continue
        seen.add(trip)
        by_state.setdefault(benign, []).append((laser, module))
    return by_state
