"""
Shared VLM inference, module analysis helpers, and state-classification metrics.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from os import PathLike
from typing import Union

import cv2
import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from helper import CANON_STATES
from ontology_common import MODULE_ORDER

# fixed mapping from RDF if it is not provided
IDX_TO_STATE = {i: s for i, s in enumerate(CANON_STATES)}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Per-module on-candidate brightness floors (section_module_analysis).
RED_BRIGHT_MIN = 0.06
GREEN_BRIGHT_MIN = 0.05
YELLOW_BRIGHT_MIN = 0.05

EXPECTED_BUCKET_BY_MODULE = {
    "top": "red_cluster",
    "middle": "yellow_cluster",
    "bottom": "green_cluster",
}

MODULE_ANALYSIS_METHODS = ("equal_thirds", "frame_blob")
DEFAULT_MODULE_ANALYSIS_METHOD = "frame_blob"

FRAME_BLOB_ABS_MIN = 0.05
FRAME_BLOB_REL_MULT = 2.5
FRAME_BLOB_MIN_AREA = 3
FRAME_BLOB_SCENE_PAD_RATIO = 0.5

# type alias for image input
ImageInput = Union[str, PathLike, Image.Image, np.ndarray]

# converts a PIL image to a numpy array
def rgb_image_to_array(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img) / 255.0

# estimates the dominant color name from a numpy array
def dominant_color_name(rgb_arr: np.ndarray) -> str:
    mean_rgb = rgb_arr.mean(axis=(0, 1))
    r, g, b = mean_rgb
    if r > g and r > b:
        return "yellow" if g > b else "red"
    if g > r and g > b:
        return "green"
    return "other"

# estimates the brightness from a numpy array
def estimate_brightness(rgb_arr: np.ndarray) -> float:
    r, g, b = rgb_arr[:, :, 0], rgb_arr[:, :, 1], rgb_arr[:, :, 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return float(y.mean())

# estimates the multi-spot brightness from a numpy array
def multi_spot_brightness(rgb_arr: np.ndarray, k: int = 3) -> dict:
    r, g, b = rgb_arr[:, :, 0], rgb_arr[:, :, 1], rgb_arr[:, :, 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    flat = y.flatten()
    topk = np.partition(flat, -k)[-k:]
    topk_sorted = np.sort(topk)[::-1]
    return {
        "primary": float(topk_sorted[0]),
        "second": float(topk_sorted[1]) if len(topk_sorted) > 1 else 0.0,
        "third": float(topk_sorted[2]) if len(topk_sorted) > 2 else 0.0,
    }

# calculates the vertical bounds for a module
def _vertical_bounds(h: int, idx: int) -> tuple[int, int]:
    y0 = int(round(idx * h / 3.0))
    y1 = int(round((idx + 1) * h / 3.0))
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return y0, y1

# calculates the luminance map from a numpy array
def luminance_map(rgb_arr: np.ndarray) -> np.ndarray:
    r, g, b = rgb_arr[..., 0], rgb_arr[..., 1], rgb_arr[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

# normalizes a bounding box
def _normalize_bbox(bbox) -> tuple[int, int, int, int] | None:
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

# reads a RGB frame from a path
def _read_rgb_frame(path: str) -> np.ndarray:
    """Load a full RGB frame as float32 in [0, 1] without caching."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        from helper import load_dtld_color

        rgb = load_dtld_color(path).astype(np.float32)
        if rgb.max() > 1.0:
            rgb /= 255.0
        return rgb
    return rgb_image_to_array(Image.open(path))

# calculates the scene brightness cache key
def _scene_brightness_cache_key(
    frame_path: str,
    bbox: tuple[int, int, int, int],
) -> tuple[str, tuple[int, int, int, int]]:
    return frame_path, bbox

# estimates the scene brightness
def estimate_scene_brightness(
    frame_path: str,
    bbox,
    scene_brightness_cache: dict | None = None,
    pad_ratio: float = FRAME_BLOB_SCENE_PAD_RATIO,
) -> float:
    """Estimate scene brightness for a bbox, caching only the scalar result."""
    norm_bbox = _normalize_bbox(bbox)
    if norm_bbox is None:
        raise ValueError(f"Invalid bbox for scene brightness: {bbox!r}")

    cache_key = _scene_brightness_cache_key(frame_path, norm_bbox)
    if scene_brightness_cache is not None and cache_key in scene_brightness_cache:
        return scene_brightness_cache[cache_key]

    frame_rgb = _read_rgb_frame(frame_path)
    try:
        value = float(scene_brightness_background_ring(frame_rgb, norm_bbox, pad_ratio=pad_ratio))
    finally:
        del frame_rgb

    if scene_brightness_cache is not None:
        scene_brightness_cache[cache_key] = value
    return value

# estimates the scene brightness from a background ring
def scene_brightness_background_ring(
    frame_rgb: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad_ratio: float = FRAME_BLOB_SCENE_PAD_RATIO,
) -> float:
    """Estimate local scene brightness from a ring around the traffic-light bbox."""
    x1, y1, x2, y2 = bbox
    h, w = frame_rgb.shape[:2]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x = max(2, int(bw * pad_ratio))
    pad_y = max(2, int(bh * pad_ratio))
    ex1, ey1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    ex2, ey2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

    patch = frame_rgb[ey1:ey2, ex1:ex2]
    y_patch = luminance_map(patch)
    ix1, iy1 = x1 - ex1, y1 - ey1
    ix2, iy2 = x2 - ex1, y2 - ey1
    mask = np.ones(y_patch.shape, dtype=bool)
    mask[iy1:iy2, ix1:ix2] = False
    ring = y_patch[mask]
    if ring.size == 0:
        return float(np.median(luminance_map(frame_rgb)))
    return float(np.median(ring))

# calculates the slot for a center y coordinate 
def _slot_for_cy(cy: float, h: int) -> str:
    if cy < h / 3.0:
        return "top"
    if cy < 2 * h / 3.0:
        return "middle"
    return "bottom"

# finalizes the module analysis
def _finalize_module_analysis(
    sections: list[dict],
    on_modules: list[str],
    *,
    method: str,
    extra: dict | None = None,
) -> dict:
    bright_values = [float(s["brightness"]) for s in sections]
    bright_sorted = sorted(bright_values, reverse=True) if bright_values else [0.0]
    primary = bright_sorted[0]
    second = bright_sorted[1] if len(bright_sorted) > 1 else 0.0
    third = bright_sorted[2] if len(bright_sorted) > 2 else 0.0
    result = {
        "method": method,
        "sections": sections,
        "num_on_modules": int(len(on_modules)),
        "on_modules": list(on_modules),
        "brightness_delta_1_2": float(primary - second),
        "brightness_delta_2_3": float(second - third),
        "brightness_spread": float(primary - third),
        "primary": float(primary),
        "second": float(second),
        "third": float(third),
    }
    if extra:
        result.update(extra)
    return result

# analyzes the modules in a section
def section_module_analysis(rgb_arr: np.ndarray) -> dict:
    """Analyze top/middle/bottom modules with per-section CIE/bucket/brightness."""
    h = int(rgb_arr.shape[0])
    sections = []
    bright_values = []

    for idx, name in enumerate(MODULE_ORDER):
        y0, y1 = _vertical_bounds(h, idx)
        sec = rgb_arr[y0:y1, :, :]
        bright = estimate_brightness(sec)
        cie = estimate_cie_x(sec)
        bucket = cie_x_bucket(cie)
        dom = dominant_color_name(sec)
        expected_bucket = EXPECTED_BUCKET_BY_MODULE[name]
        expected_on_thr = (
            RED_BRIGHT_MIN if expected_bucket == "red_cluster"
            else YELLOW_BRIGHT_MIN if expected_bucket == "yellow_cluster"
            else GREEN_BRIGHT_MIN
        )
        expected_color_match = bucket == expected_bucket
        on_candidate = expected_color_match and bright >= expected_on_thr
        bright_values.append(bright)
        sections.append({
            "name": name,
            "y0": y0,
            "y1": y1,
            "dom": dom,
            "cie_x": cie,
            "cie_bucket": bucket,
            "brightness": bright,
            "expected_bucket": expected_bucket,
            "expected_color_match": expected_color_match,
            "expected_on_threshold": expected_on_thr,
            "on_candidate": on_candidate,
        })

    bright_sorted = sorted(bright_values, reverse=True)
    on_modules = [s["name"] for s in sections if s["on_candidate"]]
    return _finalize_module_analysis(sections, on_modules, method="equal_thirds")

# analyzes the modules in a frame blob
def frame_blob_module_analysis(
    rgb_arr: np.ndarray,
    scene_brightness: float,
    *,
    abs_min: float = FRAME_BLOB_ABS_MIN,
    rel_mult: float = FRAME_BLOB_REL_MULT,
    min_area: int = FRAME_BLOB_MIN_AREA,
) -> dict:
    """Detect ON modules via scene-relative thresholding and bright blob clustering."""
    h, w = rgb_arr.shape[:2]
    y_map = luminance_map(rgb_arr)
    threshold = max(abs_min, scene_brightness * rel_mult)
    mask = (y_map >= threshold).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8,
    )

    blobs = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[label_id]
        blob_mask = labels == label_id
        blob_rgb = rgb_arr[blob_mask]
        mean_bright = float(y_map[blob_mask].mean())
        blobs.append({
            "label_id": int(label_id),
            "area": area,
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "mean_brightness": mean_bright,
            "dom": dominant_color_name(blob_rgb.reshape(-1, 1, 3)),
            "cie_x": estimate_cie_x(blob_rgb.reshape(-1, 1, 3)),
            "cie_bucket": cie_x_bucket(estimate_cie_x(blob_rgb.reshape(-1, 1, 3))),
        })

    blobs.sort(key=lambda b: b["mean_brightness"], reverse=True)

    slot_best: dict[str, dict] = {}
    for blob in blobs:
        slot = _slot_for_cy(blob["centroid_y"], h)
        prev = slot_best.get(slot)
        if prev is None or blob["mean_brightness"] > prev["mean_brightness"]:
            slot_best[slot] = {**blob, "slot": slot}

    on_modules = [slot for slot in MODULE_ORDER if slot in slot_best]
    sections = []
    for name in MODULE_ORDER:
        blob = slot_best.get(name)
        if blob is not None:
            sections.append({
                "name": name,
                "y0": int(max(0, round(blob["centroid_y"] - 1))),
                "y1": int(min(h, round(blob["centroid_y"] + 2))),
                "dom": blob["dom"],
                "cie_x": blob["cie_x"],
                "cie_bucket": blob["cie_bucket"],
                "brightness": blob["mean_brightness"],
                "blob_area": blob["area"],
                "centroid_x": blob["centroid_x"],
                "centroid_y": blob["centroid_y"],
                "on_candidate": True,
            })
        else:
            y0, y1 = _vertical_bounds(h, MODULE_ORDER.index(name))
            sec = rgb_arr[y0:y1, :, :]
            sections.append({
                "name": name,
                "y0": y0,
                "y1": y1,
                "dom": dominant_color_name(sec),
                "cie_x": estimate_cie_x(sec),
                "cie_bucket": cie_x_bucket(estimate_cie_x(sec)),
                "brightness": estimate_brightness(sec),
                "blob_area": 0,
                "centroid_x": None,
                "centroid_y": None,
                "on_candidate": False,
            })

    return _finalize_module_analysis(
        sections,
        on_modules,
        method="frame_blob",
        extra={
            "scene_brightness": float(scene_brightness),
            "brightness_threshold": float(threshold),
        },
    )

# analyzes the modules in an image
def analyze_modules(
    rgb_arr: np.ndarray,
    method: str = DEFAULT_MODULE_ANALYSIS_METHOD,
    frame_path: str | None = None,
    bbox=None,
    scene_brightness: float | None = None,
    scene_brightness_cache: dict | None = None,
) -> dict:
    if method == "equal_thirds":
        return section_module_analysis(rgb_arr)

    if method == "frame_blob":
        if scene_brightness is None:
            norm_bbox = _normalize_bbox(bbox)
            if frame_path and norm_bbox:
                scene_brightness = estimate_scene_brightness(
                    frame_path,
                    norm_bbox,
                    scene_brightness_cache=scene_brightness_cache,
                )
            else:
                scene_brightness = float(np.median(luminance_map(rgb_arr)))
        return frame_blob_module_analysis(rgb_arr, scene_brightness)

    raise ValueError(
        f"Unknown module analysis method: {method!r}. "
        f"Choose from {MODULE_ANALYSIS_METHODS}."
    )

# estimates the CIE x coordinate from a numpy array
def estimate_cie_x(rgb_arr: np.ndarray) -> float:
    mean_rgb = rgb_arr.mean(axis=(0, 1))
    r, g, b = mean_rgb
    return float(r / (r + g + b + 1e-6))

# calculates the CIE x bucket from a CIE x coordinate
def cie_x_bucket(cie_x: float) -> str:
    if cie_x > 0.60:
        return "red_cluster"
    if cie_x > 0.45:
        return "yellow_cluster"
    if cie_x > 0.30:
        return "green_cluster"
    return "other_cluster"


def _coerce_image(image: ImageInput) -> tuple[Image.Image, np.ndarray]:
    if isinstance(image, (str, PathLike)):
        img_pil = Image.open(os.fspath(image))
        rgb_arr = rgb_image_to_array(img_pil)
        return img_pil, rgb_arr

    if isinstance(image, Image.Image):
        img_pil = image.convert("RGB") if image.mode != "RGB" else image
        return img_pil, np.array(img_pil) / 255.0

    if isinstance(image, np.ndarray):
        if image.ndim < 2 or image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"Cannot infer on empty crop with shape {image.shape}")
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(arr)
        return img_pil, arr.astype(np.float32) / 255.0

    raise TypeError(f"Unsupported image input type: {type(image).__name__}")

# loads a batch of images
def load_batch_images(images: list) -> tuple[list[Image.Image], list[np.ndarray]]:
    """Load a list of paths / PIL / ndarray inputs into PIL + float RGB arrays."""
    pils = []
    rgb_arrs = []
    for image in images:
        img_pil, rgb_arr = _coerce_image(image)
        pils.append(img_pil)
        rgb_arrs.append(rgb_arr)
    return pils, rgb_arrs

# loads a batch of images with timing
def _load_batch_images_timed(images: list) -> tuple[list[Image.Image], list[np.ndarray], float]:
    t0 = time.perf_counter()
    pils, rgb_arrs = load_batch_images(images)
    return pils, rgb_arrs, time.perf_counter() - t0

# prefetches a batch of images
class BatchImagePrefetcher:
    """Prefetch the next batch's crops on a background thread during inference."""

    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._future = None

    def submit(self, images: list) -> None:
        if self._future is not None:
            raise RuntimeError("BatchImagePrefetcher: previous batch still pending")
        self._future = self._executor.submit(_load_batch_images_timed, list(images))

    def get(self) -> tuple[list[Image.Image], list[np.ndarray], float]:
        if self._future is None:
            raise RuntimeError("BatchImagePrefetcher: no batch submitted")
        pils, rgb_arrs, load_s = self._future.result()
        self._future = None
        return pils, rgb_arrs, load_s

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            # Python < 3.9
            self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

# loads a model and classifier
def load_model_and_classifier(model_path: str):
    """Load CLIP visual encoder + classifier from a VLM checkpoint."""
    clip_full, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-32", pretrained=None,
    )
    visual = clip_full.visual
    feature_dim = visual.output_dim

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    num_classes = len(CANON_STATES)
    for k, v in checkpoint.items():
        if k.endswith("classifier.weight"):
            num_classes = v.shape[0]
            break

    classifier = nn.Linear(feature_dim, num_classes)
    visual_state, clf_state = {}, {}
    for k, v in checkpoint.items():
        if k.startswith("model.visual."):
            visual_state[k[len("model.visual."):]] = v
        elif k.startswith("classifier."):
            clf_state[k[len("classifier."):]] = v
        elif k.endswith("classifier.weight") or k.endswith("classifier.bias"):
            clf_state[k.split("classifier.")[-1]] = v

    visual.load_state_dict(visual_state, strict=False)
    classifier.load_state_dict(clf_state, strict=False)
    visual.to(DEVICE).eval()
    classifier.to(DEVICE).eval()
    return visual, classifier, preprocess, num_classes

# predicts the state of an image
def predict(
    model,
    classifier,
    preprocess,
    image: ImageInput,
    num_classes: int | None = None,
    *,
    module_analysis_method: str | None = None,
    frame_path: str | None = None,
    bbox=None,
    scene_brightness: float | None = None,
    scene_brightness_cache: dict | None = None,
    timings_out: dict | None = None,
) -> dict:
    """Run VLM inference on a file path, PIL image, or uint8 RGB ndarray.

    If timings_out is provided, it is filled with seconds spent in:
    image_load, inference, color_features, module_analysis.
    """
    results = predict_batch(
        model,
        classifier,
        preprocess,
        [image],
        num_classes,
        module_analysis_method=module_analysis_method,
        frame_paths=[frame_path],
        bboxes=[bbox],
        scene_brightness_values=[scene_brightness],
        scene_brightness_cache=scene_brightness_cache,
        timings_out=timings_out,
    )
    return results[0]


# predicts the state of a batch of images
def predict_batch(
    model,
    classifier,
    preprocess,
    images: list,
    num_classes: int | None = None,
    *,
    module_analysis_method: str | None = None,
    frame_paths: list | None = None,
    bboxes: list | None = None,
    scene_brightness_values: list | None = None,
    scene_brightness_cache: dict | None = None,
    timings_out: dict | None = None,
    preloaded: tuple[list[Image.Image], list[np.ndarray]] | None = None,
    image_load_s: float | None = None,
) -> list[dict]:
    """Batched VLM inference over file paths / PIL images / RGB ndarrays."""
    if not images:
        if timings_out is not None:
            for key in ("image_load", "inference", "color_features", "module_analysis"):
                timings_out[key] = timings_out.get(key, 0.0)
        return []

    n = len(images)
    frame_paths = list(frame_paths) if frame_paths is not None else [None] * n
    bboxes = list(bboxes) if bboxes is not None else [None] * n
    scene_brightness_values = (
        list(scene_brightness_values)
        if scene_brightness_values is not None
        else [None] * n
    )
    if not (len(frame_paths) == len(bboxes) == len(scene_brightness_values) == n):
        raise ValueError(
            "frame_paths, bboxes, and scene_brightness_values must match images length"
        )

    if preloaded is not None:
        pils, rgb_arrs = preloaded
        if len(pils) != n or len(rgb_arrs) != n:
            raise ValueError("preloaded batch size must match images length")
        load_elapsed = image_load_s if image_load_s is not None else 0.0
    else:
        t0 = time.perf_counter()
        pils, rgb_arrs = load_batch_images(images)
        load_elapsed = time.perf_counter() - t0

    t_infer_start = time.perf_counter()

    with torch.no_grad():
        batch = torch.stack([preprocess(img) for img in pils], dim=0).to(DEVICE)
        feats = model(batch)
        logits = classifier(feats)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        logits_np = logits.cpu().numpy()
    t_infer = time.perf_counter()

    if num_classes is None:
        num_classes = int(probs.shape[1])

    results = []
    t_color_total = 0.0
    t_module_total = 0.0
    for i in range(n):
        t_c0 = time.perf_counter()
        row_probs = probs[i]
        pred_idx = int(np.argmax(row_probs))
        state = IDX_TO_STATE.get(pred_idx, f"unknown_{pred_idx}")
        state_probs = {
            IDX_TO_STATE.get(j, f"unknown_{j}"): float(row_probs[j])
            for j in range(min(num_classes, len(row_probs)))
        }
        rgb_arr = rgb_arrs[i]
        cie = estimate_cie_x(rgb_arr)
        multi = multi_spot_brightness(rgb_arr)
        dom = dominant_color_name(rgb_arr)
        bright = estimate_brightness(rgb_arr)
        t_c1 = time.perf_counter()
        t_color_total += t_c1 - t_c0

        result = {
            "state": state,
            "conf": float(row_probs[pred_idx]),
            "state_probs": state_probs,
            "probs": state_probs,
            "logits": logits_np[i].tolist(),
            "dom": dom,
            "bright": bright,
            "multi": multi,
            "cie_x": cie,
            "cie_bucket": cie_x_bucket(cie),
        }

        if module_analysis_method is not None:
            module_analysis = analyze_modules(
                rgb_arr,
                method=module_analysis_method,
                frame_path=frame_paths[i],
                bbox=bboxes[i],
                scene_brightness=scene_brightness_values[i],
                scene_brightness_cache=scene_brightness_cache,
            )
            result["module_analysis"] = module_analysis
            result["module_sections"] = module_analysis["sections"]
        t_module_total += time.perf_counter() - t_c1
        results.append(result)

    if timings_out is not None:
        timings_out["image_load"] = timings_out.get("image_load", 0.0) + load_elapsed
        timings_out["inference"] = timings_out.get("inference", 0.0) + (t_infer - t_infer_start)
        timings_out["color_features"] = timings_out.get("color_features", 0.0) + t_color_total
        timings_out["module_analysis"] = timings_out.get("module_analysis", 0.0) + t_module_total

    return results

# predicts the state of an image from a file path
def predict_state(model, classifier, preprocess, img_path, num_classes, **kwargs):
    """Backward-compatible wrapper: predict from an image file path."""
    return predict(model, classifier, preprocess, img_path, num_classes, **kwargs)

# predicts the state of a crop
def predict_from_crop(model, classifier, preprocess, crop_arr: np.ndarray, num_classes=None, **kwargs):
    """Backward-compatible wrapper: predict from a uint8 RGB crop array."""
    return predict(model, classifier, preprocess, crop_arr, num_classes, **kwargs)

# counts the true and false positives and negatives for a label
def _one_vs_rest_counts(y_true, y_pred, label):
    tp = fp = tn = fn = 0
    for gt, pred in zip(y_true, y_pred):
        gt_pos = gt == label
        pred_pos = pred == label
        if gt_pos and pred_pos:
            tp += 1
        elif not gt_pos and pred_pos:
            fp += 1
        elif gt_pos and not pred_pos:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn

# computes the metrics for a subset of predictions
def compute_state_metrics(rows: list, variant: str | None = None) -> dict:
    if variant is not None:
        subset = [r for r in rows if r["variant"] == variant]
    else:
        subset = rows

    empty = {
        "total": 0,
        "accuracy": 0.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
        "macro_f1": 0.0,
        "weighted_precision": 0.0,
        "weighted_recall": 0.0,
        "weighted_f1": 0.0,
        "per_class": {},
    }
    if not subset:
        return empty

    y_true = [r["gt_state"] for r in subset]
    y_pred = [r["pred_state"] for r in subset]
    labels = sorted(set(y_true))

    macro_kw = dict(average="macro", zero_division=0, labels=labels)
    weighted_kw = dict(average="weighted", zero_division=0, labels=labels)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )

    per_class = {}
    for i, label in enumerate(labels):
        tp, fp, tn, fn = _one_vs_rest_counts(y_true, y_pred, label)
        per_class[label] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(support[i]),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    return {
        "total": len(subset),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, **macro_kw)),
        "macro_recall": float(recall_score(y_true, y_pred, **macro_kw)),
        "macro_f1": float(f1_score(y_true, y_pred, **macro_kw)),
        "weighted_precision": float(precision_score(y_true, y_pred, **weighted_kw)),
        "weighted_recall": float(recall_score(y_true, y_pred, **weighted_kw)),
        "weighted_f1": float(f1_score(y_true, y_pred, **weighted_kw)),
        "per_class": per_class,
    }

# prints the metrics for a subset of predictions
def print_state_metrics_block(title: str, metrics: dict):
    print(f"\n--- {title} (n={metrics['total']}) ---")
    if not metrics["total"]:
        print("  no samples")
        return
    print(
        f"accuracy={metrics['accuracy'] * 100:.2f}%  "
        f"macro_p={metrics['macro_precision']:.3f}  "
        f"macro_r={metrics['macro_recall']:.3f}  "
        f"macro_f1={metrics['macro_f1']:.3f}  "
        f"weighted_f1={metrics['weighted_f1']:.3f}"
    )
    for cls, v in sorted(metrics["per_class"].items()):
        print(
            f"  {cls}: p={v['precision']:.3f} r={v['recall']:.3f} "
            f"f1={v['f1']:.3f} n={v['support']}  "
            f"TP={v['tp']} FP={v['fp']} TN={v['tn']} FN={v['fn']}"
        )

# prints the comparison of models
def print_comparison(summaries: dict):
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON")
    print(f"{'=' * 60}")

    # Prefer a stable column order when present.
    preferred = ("baseline", "ontology", "ontology_2")
    tags = [t for t in preferred if t in summaries]
    tags.extend(t for t in summaries if t not in tags)
    if not tags:
        print("  (no model summaries)")
        return

    col_w = max(12, max(len(t) for t in tags) + 2)
    header = f"{'Metric':<28}" + "".join(f"{t:>{col_w}}" for t in tags)
    print(header)
    print("-" * len(header))

    metrics = [
        ("state macro f1 (all)", "state_metrics", "{:.3f}", "macro_f1"),
        ("state macro f1 (benign)", "benign_state_metrics", "{:.3f}", "macro_f1"),
        ("state macro f1 (attack)", "attack_state_metrics", "{:.3f}", "macro_f1"),
        ("state accuracy (all)", "state_metrics", "{:.2%}", "accuracy"),
        ("state accuracy (benign)", "benign_state_metrics", "{:.2%}", "accuracy"),
        ("state accuracy (attack)", "attack_state_metrics", "{:.2%}", "accuracy"),
        ("attack success rate", "attack_state_success_rate", "{:.2%}"),
    ]
    for item in metrics:
        label, key, fmt = item[0], item[1], item[2]
        subkey = item[3] if len(item) > 3 else None
        cells = []
        for tag in tags:
            summary = summaries.get(tag, {})
            if subkey:
                val = summary.get(key, {}).get(subkey, 0)
            else:
                val = summary.get(key, 0)
            cells.append(fmt.format(val))
        print(f"{label:<28}" + "".join(f"{c:>{col_w}}" for c in cells))
