"""
Train baseline or card_owned ontology models on DTLD crops.

  baseline    — CLIP classification CE only (no aux losses)
  card_owned  — cardinality-owned ontology model (aux losses)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import open_clip
except Exception:  # pragma: no cover
    open_clip = None

# helper functions for the DTLD dataset
from helper import (
    DEFAULT_ATTR_FILTERS,
    load_attr_filters,
    load_dtld_color,
    resolve_dtld_image_path,
    scan_filtered_crop_records,
)
# shared ontology / module helpers
from ontology_common import (
    BOTTOM,
    CANON_STATES,
    CARDINALITY_STATES,
    CD_MAP_OFF_TARGET,
    CD_MAP_ON_TARGET,
    DEFAULT_MODULE_INTENSITY_RANGES,
    IDEAL_XY,
    MIDDLE,
    MODULE_IDX,
    MODULE_ON_INTENSITY_THRESHOLD,
    MODULE_ORDER,
    MODULE_STATE_NAMES,
    MODULE_STATE_TO_IDX,
    NUM_CLASSES,
    NUM_MODULES,
    NUM_MODULE_STATES,
    ONTOLOGY_MODULE_SPECS,
    STATE_ON_MASK,
    STATE_TO_IDX,
    TOP,
    TRAFFIC_COLORS,
    CdMap,
    color_xy_from_module_feats,
    module_intensity_from_feats,
    module_states_from_light_state,
    pil_to_gray_rgb,
    split_module_pils,
    split_three_zones,
)

ATTRIBUTE_VALUE_MAPS: dict[str, dict[str, int]] = {
    "isControlledBy": {
        "TopModule": 0,
        "MiddleModule": 1,
        "BottomModule": 2,
        "Top and Middle Module": 3,
        "None": 4,
    },
    "hasCIE_x": {
        "0.67 to 0.70": 0,
        "0.55 to 0.58": 1,
        "0.15 to 0.20": 2,
        "Blended": 3,
        "N/A": 4,
    },
    "hasDominantWavelength": {
        "610nm to 625nm": 0,
        "590nm to 595nm": 1,
        "505nm to 520nm": 2,
        "Mixed": 3,
        "N/A": 4,
    },
}
NUM_ONTOLOGY_ELEMENTS = len(ATTRIBUTE_VALUE_MAPS)

ONTOLOGY_STATE_MAPPINGS: dict[str, dict[str, str]] = {
    "red": {
        "isControlledBy": "TopModule",
        "hasCIE_x": "0.67 to 0.70",
        "hasDominantWavelength": "610nm to 625nm",
    },
    "yellow": {
        "isControlledBy": "MiddleModule",
        "hasCIE_x": "0.55 to 0.58",
        "hasDominantWavelength": "590nm to 595nm",
    },
    "green": {
        "isControlledBy": "BottomModule",
        "hasCIE_x": "0.15 to 0.20",
        "hasDominantWavelength": "505nm to 520nm",
    },
    "red_yellow": {
        "isControlledBy": "Top and Middle Module",
        "hasCIE_x": "Blended",
        "hasDominantWavelength": "Mixed",
    },
    "off": {
        "isControlledBy": "None",
        "hasCIE_x": "N/A",
        "hasDominantWavelength": "N/A",
    },
}

STATE_DESCRIPTIONS = {
    "red": "The signal shows red.",
    "yellow": "The signal shows yellow.",
    "green": "The signal shows green.",
    "red_yellow": "The signal shows red and yellow.",
    "off": "The signal is off or shows no state.",
}
DEFAULT_INTENSITY_ON_TEMPERATURE = 10.0

# default card_owned recipe weights
DEFAULT_CARD_OWNED_WEIGHTS = {
    "contrastive": 0.262,
    "attribute": 0.19,
    "color_method": 0.38,
    "module_state": 0.38,
    "module_intensity": 0.38,
    "cardinality": 0.38,
    "brightness_rank": 0.38,
}


# brightness helpers (zone / ring ratios + ranking)


# mean brightness of a zone image
def _zone_mean_brightness(zone: np.ndarray) -> float:
    if zone.size == 0:
        return float("nan")
    if zone.ndim == 2:
        gray = zone.astype(np.float32)
    else:
        gray = cv2.cvtColor(zone, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(gray.mean())


# top/middle/bottom zone brightnesses
def _zone_brightness_triplet(roi_rgb: np.ndarray) -> np.ndarray:
    zones = split_three_zones(roi_rgb)
    return np.array(
        [_zone_mean_brightness(zones[name]) for name in MODULE_ORDER],
        dtype=np.float32,
    )


# clamps an xywh box to image bounds
def _clamp_box(x, y, w, h, img_w, img_h) -> tuple[int, int, int, int]:
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(img_w, int(round(x + w)))
    y2 = min(img_h, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, 0, 0
    return x1, y1, x2, y2


# crops a roi and a larger context border
def _crop_with_border(img, x, y, w, h, border: int = 20):
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = _clamp_box(x, y, w, h, img_w, img_h)
    if x2 <= x1 or y2 <= y1:
        empty = img[0:0, 0:0]
        return empty, empty, (x1, y1, x2, y2), (x1, y1, x2, y2)
    bw = (x2 - x1) + 2 * border
    bh = (y2 - y1) + 2 * border
    bx1, by1, bx2, by2 = _clamp_box(x1 - border, y1 - border, bw, bh, img_w, img_h)
    roi = img[y1:y2, x1:x2]
    context = img[by1:by2, bx1:bx2]
    return roi, context, (x1, y1, x2, y2), (bx1, by1, bx2, by2)


# mean brightness of the ring around a bbox
def _ring_mean_brightness(img_rgb: np.ndarray, x, y, w, h, border: int = 20) -> float:
    roi, context, box, ctx = _crop_with_border(img_rgb, x, y, w, h, border=border)
    if roi.size == 0 or context.size == 0:
        return float("nan")
    gray_ctx = cv2.cvtColor(context, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ctx_mask = np.ones(gray_ctx.shape, dtype=bool)
    rx1 = box[0] - ctx[0]
    ry1 = box[1] - ctx[1]
    rx2 = rx1 + roi.shape[1]
    ry2 = ry1 + roi.shape[0]
    ctx_mask[ry1:ry2, rx1:rx2] = False
    ring = gray_ctx[ctx_mask]
    if ring.size == 0:
        return float("nan")
    return float(ring.mean())


# zone brightness divided by ring brightness
def zone_ring_ratios(
    roi_rgb: np.ndarray,
    frame_rgb: np.ndarray,
    bbox_xywh: Sequence[float],
    border: int = 20,
    eps: float = 1e-6,
) -> np.ndarray | None:
    if roi_rgb is None or roi_rgb.size == 0:
        return None
    zones = _zone_brightness_triplet(roi_rgb)
    if not np.all(np.isfinite(zones)):
        return None
    x, y, w, h = bbox_xywh
    ring = _ring_mean_brightness(frame_rgb, x, y, w, h, border=border)
    if not np.isfinite(ring):
        return None
    return (zones / (ring + eps)).astype(np.float32)


# loads a full-frame RGB image from disk
def _load_frame_rgb(
    frame_path: str,
    dtld_root: str | None,
    fast_tiff: bool = True,
) -> np.ndarray | None:
    path = frame_path
    if not path:
        return None
    if not os.path.isfile(path) and dtld_root:
        resolved = resolve_dtld_image_path(path, dtld_root)
        if resolved:
            path = resolved
    if not os.path.isfile(path):
        return None
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".tif", ".tiff"):
            return load_dtld_color(path, use_stretch=not fast_tiff)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


# loads a brightness-target JSON cache
def _load_brightness_cache(cache_path: str) -> dict[str, dict[str, Any]]:
    if not cache_path or not os.path.isfile(cache_path):
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    return data if isinstance(data, dict) else {}


# writes a brightness-target JSON cache
def _save_brightness_cache(
    cache_path: str,
    records: list[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> None:
    entries = {}
    for rec in records:
        fn = rec.get("filename")
        if not fn:
            continue
        entries[str(fn)] = {
            "brightness_targets": rec.get("brightness_targets", [0.0, 0.0, 0.0]),
            "brightness_valid": bool(rec.get("brightness_valid", False)),
        }
    payload = {"meta": meta or {}, "entries": entries}
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


# applies cached brightness targets onto records
def _apply_brightness_cache(
    records: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    hit = miss = 0
    for rec in records:
        fn = str(rec.get("filename") or "")
        entry = cache.get(fn)
        if entry is None:
            miss += 1
            continue
        rec["brightness_targets"] = entry.get("brightness_targets", [0.0, 0.0, 0.0])
        rec["brightness_valid"] = bool(entry.get("brightness_valid", False))
        hit += 1
    return hit, miss


# precomputes zone/ring brightness ratios for records
def enrich_records_with_zone_ring_ratios(
    records: list[dict[str, Any]],
    image_dir: str,
    dtld_root: str | None = None,
    border: int = 20,
    cache_path: str | None = None,
    fast_tiff: bool = True,
    show_progress: bool = True,
) -> dict[str, int]:
    """Precompute zone/ring brightness ratios; optional JSON cache."""
    stats = {
        "total": len(records),
        "valid": 0,
        "missing_frame": 0,
        "missing_bbox": 0,
        "bad_crop": 0,
        "bad_ratio": 0,
        "cache_hits": 0,
        "computed": 0,
    }

    cache = _load_brightness_cache(cache_path) if cache_path else {}
    if cache:
        hit, miss = _apply_brightness_cache(records, cache)
        stats["cache_hits"] = hit
        if miss == 0:
            stats["valid"] = sum(1 for r in records if r.get("brightness_valid"))
            return stats
        todo_indices = [
            i for i, r in enumerate(records) if str(r.get("filename") or "") not in cache
        ]
    else:
        todo_indices = list(range(len(records)))

    by_frame: dict[str, list[int]] = defaultdict(list)
    for i in todo_indices:
        by_frame[str(records[i].get("frame_image_path") or "")].append(i)

    frame_items = list(by_frame.items())
    iterator = frame_items
    if show_progress:
        try:
            iterator = tqdm(
                frame_items,
                desc="Brightness targets",
                leave=True,
                dynamic_ncols=True,
                unit="frame",
            )
        except Exception:
            pass

    for frame_path, idxs in iterator:
        frame_rgb = (
            _load_frame_rgb(frame_path, dtld_root, fast_tiff=fast_tiff)
            if frame_path
            else None
        )
        if frame_rgb is None:
            for i in idxs:
                records[i]["brightness_targets"] = [0.0, 0.0, 0.0]
                records[i]["brightness_valid"] = False
                stats["missing_frame"] += 1
            continue

        for i in idxs:
            rec = records[i]
            bbox = rec.get("bbox_xywh")
            if not bbox or len(bbox) != 4:
                rec["brightness_targets"] = [0.0, 0.0, 0.0]
                rec["brightness_valid"] = False
                stats["missing_bbox"] += 1
                continue

            crop_path = os.path.join(image_dir, rec["filename"])
            try:
                crop = np.array(Image.open(crop_path).convert("RGB"))
            except Exception:
                rec["brightness_targets"] = [0.0, 0.0, 0.0]
                rec["brightness_valid"] = False
                stats["bad_crop"] += 1
                continue

            ratios = zone_ring_ratios(crop, frame_rgb, bbox, border=border)
            if ratios is None:
                rec["brightness_targets"] = [0.0, 0.0, 0.0]
                rec["brightness_valid"] = False
                stats["bad_ratio"] += 1
                continue

            rec["brightness_targets"] = ratios.tolist()
            rec["brightness_valid"] = True
            stats["valid"] += 1
            stats["computed"] += 1

        del frame_rgb

    stats["valid"] = sum(1 for r in records if r.get("brightness_valid"))
    if cache_path:
        _save_brightness_cache(
            cache_path,
            records,
            meta={
                "image_dir": image_dir,
                "dtld_root": dtld_root,
                "border": border,
                "fast_tiff": fast_tiff,
                "n_records": len(records),
            },
        )
    return stats


# state-conditioned hinge ranking on zone/ring ratios (loss)
def module_brightness_ranking_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    canon_states: Sequence[str],
    margin: float = 0.05,
    off_margin: float = 0.05,
) -> torch.Tensor:
    """State-conditioned hinge ranking on predicted zone/ring ratios (B, 3)."""
    if pred.numel() == 0:
        return pred.new_zeros(())
    per_sample = []
    for i in range(pred.size(0)):
        state = canon_states[int(labels[i].item())]
        t, m, b = pred[i, TOP], pred[i, MIDDLE], pred[i, BOTTOM]
        if state == "red":
            loss_i = F.relu(margin - (t - m)) + F.relu(margin - (t - b))
        elif state == "yellow":
            loss_i = F.relu(margin - (m - t)) + F.relu(margin - (m - b))
        elif state == "green":
            loss_i = F.relu(margin - (b - t)) + F.relu(margin - (b - m))
        elif state == "red_yellow":
            loss_i = F.relu(margin - (t - b)) + F.relu(margin - (m - b))
        elif state == "off":
            spread = torch.max(torch.stack([t, m, b])) - torch.min(torch.stack([t, m, b]))
            loss_i = F.relu(spread - off_margin)
        else:
            loss_i = pred.new_zeros(())
        per_sample.append(loss_i)
    return torch.stack(per_sample).mean()


# module helpers (state/intensity losses)


# fits the cd map from measured zone/ring ratios
def fit_cd_map_from_records(
    records: Sequence[dict[str, Any]],
    canon_states: Sequence[str],
    *,
    off_cd: float = CD_MAP_OFF_TARGET,
    on_cd: float = CD_MAP_ON_TARGET,
) -> CdMap:
    """Fit r->cd using measured zone/ring ratios and ontology on/off slots."""
    off_vals: list[float] = []
    on_vals: list[float] = []
    for rec in records:
        if not rec.get("brightness_valid"):
            continue
        r = rec.get("brightness_targets") or [0.0, 0.0, 0.0]
        if len(r) != 3:
            continue
        state = str(rec.get("state") or "")
        if state not in canon_states:
            continue
        mods = module_states_from_light_state(state)
        for i, m in enumerate(mods):
            val = float(r[i])
            if not (val == val):
                continue
            if m == "off":
                off_vals.append(val)
            else:
                on_vals.append(val)

    if not off_vals or not on_vals:
        return CdMap(a=50.0, b=50.0, r_off_mean=0.0, r_on_mean=2.0, gate_threshold=1.0)

    r_off = sum(off_vals) / len(off_vals)
    r_on = sum(on_vals) / len(on_vals)
    denom = (r_on - r_off) if abs(r_on - r_off) > 1e-6 else 1e-6
    b = (float(on_cd) - float(off_cd)) / denom
    a = float(off_cd) - b * r_off
    gate = 0.5 * (r_off + r_on)
    return CdMap(a=a, b=b, r_off_mean=r_off, r_on_mean=r_on, gate_threshold=gate)


# builds per-module state targets from light labels
def module_state_targets_from_labels(
    labels: torch.Tensor,
    canon_states: Sequence[str],
) -> torch.Tensor:
    device = labels.device
    B = labels.size(0)
    targets = torch.zeros(B, NUM_MODULES, device=device, dtype=torch.long)
    for i in range(B):
        light = canon_states[int(labels[i].item())]
        for m, name in enumerate(module_states_from_light_state(light)):
            targets[i, m] = MODULE_STATE_TO_IDX[name]
    return targets


# module-state logits from shared head features
def module_state_logits_from_feats(
    module_feats: torch.Tensor,
    module_state_head: nn.Module,
) -> torch.Tensor:
    if module_feats.dim() == 3:
        B, M, D = module_feats.shape
        flat = module_feats.reshape(B * M, D)
    else:
        flat = module_feats
        B = flat.shape[0] // NUM_MODULES
        M = NUM_MODULES
    return module_state_head(flat).reshape(B, M, NUM_MODULE_STATES)


# cross-entropy over per-module state predictions
def module_state_ce_loss(
    state_logits: torch.Tensor,
    state_targets: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        state_logits.reshape(-1, NUM_MODULE_STATES),
        state_targets.reshape(-1),
    )


# midpoints of default intensity bands per module color
def module_band_midpoints(
    ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    bands = ranges or DEFAULT_MODULE_INTENSITY_RANGES
    return {name: 0.5 * (float(lo) + float(hi)) for name, (lo, hi) in bands.items()}


# GT intensity midpoints for module state targets
def _gt_midpoint_tensor(
    state_targets: torch.Tensor,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    mids = module_band_midpoints(ranges)
    out = state_targets.new_zeros(state_targets.shape, dtype=torch.float32)
    for name, mid in mids.items():
        idx = MODULE_STATE_TO_IDX[name]
        out = torch.where(state_targets == idx, out.new_full((), float(mid)), out)
    return out


# softmax-relative intensity loss vs measured ratios
def module_softmax_rel_loss(
    raw: torch.Tensor,
    *,
    state_targets: torch.Tensor,
    brightness_r: torch.Tensor,
    brightness_valid: torch.Tensor,
    cd_map: CdMap | None,
    ranges: dict[str, tuple[float, float]] | None = None,
    softmax_T: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL(softmax(pred_r) || softmax(measured_r)) + SmoothL1 to GT midpoints.

    Returns ``(loss, intensity_for_checker)``.
    """
    pred_r = F.softplus(raw)
    gt_mid = _gt_midpoint_tensor(state_targets, ranges)
    valid = brightness_valid.bool()
    T = max(float(softmax_T), 1e-4)
    if valid.any():
        p_meas = F.softmax(brightness_r[valid] / T, dim=-1)
        p_pred = F.softmax(pred_r[valid] / T, dim=-1)
        loss_rel = F.kl_div((p_pred + 1e-8).log(), p_meas, reduction="batchmean")
    else:
        loss_rel = raw.new_zeros(())
    if cd_map is not None:
        intensity = cd_map(pred_r).clamp(min=0.0)
    else:
        intensity = pred_r.clamp(min=0.0)
    loss = loss_rel + F.smooth_l1_loss(intensity, gt_mid)
    return loss, intensity


# adaptive CIE xy boxes + NBS / box losses


# AdaptiveXyBoxes class - expand-only CIE xy boxes around ideal xy
class AdaptiveXyBoxes:
    """Expand-only asymmetric axis-aligned CIE xy boxes seeded at ideal xy."""

    def __init__(
        self,
        ideal_xy: dict[str, tuple[float, float]] | None = None,
        init_half_x: float = 0.01,
        init_half_y: float = 0.01,
        min_half: float = 0.005,
        expand_margin: float = 0.002,
        ema: float = 0.05,
        percentile: float = 95.0,
        enforce_non_overlap: bool = True,
        update_every: int = 1,
    ) -> None:
        ideal = ideal_xy or IDEAL_XY
        self.ideal = {c: (float(ideal[c][0]), float(ideal[c][1])) for c in TRAFFIC_COLORS}
        hx0, hy0 = float(init_half_x), float(init_half_y)
        self.x_lo = {c: self.ideal[c][0] - hx0 for c in TRAFFIC_COLORS}
        self.x_hi = {c: self.ideal[c][0] + hx0 for c in TRAFFIC_COLORS}
        self.y_lo = {c: self.ideal[c][1] - hy0 for c in TRAFFIC_COLORS}
        self.y_hi = {c: self.ideal[c][1] + hy0 for c in TRAFFIC_COLORS}
        self.min_half = float(min_half)
        self.min_width = 2.0 * self.min_half
        self.expand_margin = float(expand_margin)
        self.ema = float(ema)
        self.percentile = float(percentile)
        self.enforce_non_overlap = bool(enforce_non_overlap)
        self.update_every = max(1, int(update_every))
        self._steps = 0
        self._pending: dict[str, list[np.ndarray]] = {c: [] for c in TRAFFIC_COLORS}
        self.n_updates = {c: 0 for c in TRAFFIC_COLORS}
        if self.enforce_non_overlap:
            self._enforce_non_overlap()

    def _ensure_min_size(self, color: str) -> None:
        if self.x_hi[color] - self.x_lo[color] < self.min_width:
            mx = 0.5 * (self.x_lo[color] + self.x_hi[color])
            self.x_lo[color] = mx - 0.5 * self.min_width
            self.x_hi[color] = mx + 0.5 * self.min_width
        if self.y_hi[color] - self.y_lo[color] < self.min_width:
            my = 0.5 * (self.y_lo[color] + self.y_hi[color])
            self.y_lo[color] = my - 0.5 * self.min_width
            self.y_hi[color] = my + 0.5 * self.min_width

    def box(self, color: str) -> dict[str, float]:
        x_lo, x_hi = float(self.x_lo[color]), float(self.x_hi[color])
        y_lo, y_hi = float(self.y_lo[color]), float(self.y_hi[color])
        return {
            "x_lo": x_lo,
            "x_hi": x_hi,
            "y_lo": y_lo,
            "y_hi": y_hi,
            "x_mid": 0.5 * (x_lo + x_hi),
            "y_mid": 0.5 * (y_lo + y_hi),
            "half_x": 0.5 * (x_hi - x_lo),
            "half_y": 0.5 * (y_hi - y_lo),
            "ideal_x": float(self.ideal[color][0]),
            "ideal_y": float(self.ideal[color][1]),
        }

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {c: self.box(c) for c in TRAFFIC_COLORS}

    def _enforce_non_overlap(self) -> None:
        cols = list(TRAFFIC_COLORS)
        for _ in range(8):
            moved = False
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    a, b = cols[i], cols[j]
                    ba, bb = self.box(a), self.box(b)
                    overlap_x = min(ba["x_hi"], bb["x_hi"]) - max(ba["x_lo"], bb["x_lo"])
                    overlap_y = min(ba["y_hi"], bb["y_hi"]) - max(ba["y_lo"], bb["y_lo"])
                    if overlap_x <= 0 or overlap_y <= 0:
                        continue
                    if overlap_x >= overlap_y:
                        shrink = 0.5 * overlap_x + 1e-4
                        if ba["x_mid"] <= bb["x_mid"]:
                            self.x_hi[a] -= shrink
                            self.x_lo[b] += shrink
                        else:
                            self.x_hi[b] -= shrink
                            self.x_lo[a] += shrink
                    else:
                        shrink = 0.5 * overlap_y + 1e-4
                        if ba["y_mid"] <= bb["y_mid"]:
                            self.y_hi[a] -= shrink
                            self.y_lo[b] += shrink
                        else:
                            self.y_hi[b] -= shrink
                            self.y_lo[a] += shrink
                    moved = True
            for c in TRAFFIC_COLORS:
                self._ensure_min_size(c)
            if not moved:
                break

    def _update_from_xy_by_color(
        self,
        by_color: dict[str, list[np.ndarray]],
        *,
        use_ema: bool = True,
    ) -> None:
        lo_pct = max(0.0, 100.0 - float(self.percentile))
        hi_pct = float(self.percentile)
        for color, pts in by_color.items():
            if not pts:
                continue
            arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
            need_x_lo = float(np.percentile(arr[:, 0], lo_pct)) - self.expand_margin
            need_x_hi = float(np.percentile(arr[:, 0], hi_pct)) + self.expand_margin
            need_y_lo = float(np.percentile(arr[:, 1], lo_pct)) - self.expand_margin
            need_y_hi = float(np.percentile(arr[:, 1], hi_pct)) + self.expand_margin
            if need_x_hi - need_x_lo < self.min_width:
                mx = 0.5 * (need_x_lo + need_x_hi)
                need_x_lo, need_x_hi = mx - 0.5 * self.min_width, mx + 0.5 * self.min_width
            if need_y_hi - need_y_lo < self.min_width:
                my = 0.5 * (need_y_lo + need_y_hi)
                need_y_lo, need_y_hi = my - 0.5 * self.min_width, my + 0.5 * self.min_width

            old = (
                self.x_lo[color],
                self.x_hi[color],
                self.y_lo[color],
                self.y_hi[color],
            )
            if use_ema:
                blend = [
                    (1.0 - self.ema) * old[i] + self.ema * need
                    for i, need in enumerate(
                        (need_x_lo, need_x_hi, need_y_lo, need_y_hi)
                    )
                ]
            else:
                blend = [need_x_lo, need_x_hi, need_y_lo, need_y_hi]
            # expand-only per edge
            self.x_lo[color] = float(min(old[0], blend[0]))
            self.x_hi[color] = float(max(old[1], blend[1]))
            self.y_lo[color] = float(min(old[2], blend[2]))
            self.y_hi[color] = float(max(old[3], blend[3]))
            self._ensure_min_size(color)
            self.n_updates[color] += 1
        if self.enforce_non_overlap:
            self._enforce_non_overlap()

    @torch.no_grad()
    def update_from_batch(
        self,
        xy: torch.Tensor,
        labels: torch.Tensor,
        canon_states: Sequence[str],
        module_mask: torch.Tensor,
    ) -> None:
        """Collect active-module xy; expand boxes every ``update_every`` batches."""
        if xy.numel() == 0 or not module_mask.any():
            return
        pts = xy.detach().float().cpu().numpy().reshape(-1, 3, 2)
        mask = module_mask.detach().cpu().numpy() > 0.5
        labels_np = labels.detach().cpu().numpy().astype(int)

        for i in range(pts.shape[0]):
            state = canon_states[int(labels_np[i])]
            modules = ONTOLOGY_MODULE_SPECS.get(state, {})
            for module_name, spec in modules.items():
                idx = MODULE_ORDER.index(module_name)
                if not mask[i, idx]:
                    continue
                self._pending[spec["color"]].append(pts[i, idx].copy())

        self._steps += 1
        if self._steps % self.update_every != 0:
            return
        pending = self._pending
        self._pending = {c: [] for c in TRAFFIC_COLORS}
        if not any(pending[c] for c in TRAFFIC_COLORS):
            return
        self._update_from_xy_by_color(pending, use_ema=True)


# NBS handbook soft violation for a predicted xy
def cie_xy_nbs_violation(x: torch.Tensor, y: torch.Tensor, color: str) -> torch.Tensor:
    """Differentiable soft NBS Handbook 95 violation (ReLU hinges)."""
    if color == "red":
        return F.relu(0.992 - (x + y)) + F.relu(0.330 - (x - y))
    if color == "yellow":
        return (
            F.relu(0.383 - y)
            + F.relu(0.872 + 0.200 * x - (x + y))
            + F.relu(0.550 - x)
        )
    if color == "green":
        return (
            F.relu(x - (0.360 - 0.080 * y))
            + F.relu(x - 0.650 * y)
            + F.relu(0.390 - 0.171 * x - y)
        )
    raise ValueError(f"Unknown NBS color: {color}")


# soft range loss keeping xy inside adaptive boxes
def cie_xy_box_range_loss(
    pred_xy: torch.Tensor,
    labels: torch.Tensor,
    canon_states: Sequence[str],
    boxes: AdaptiveXyBoxes,
    module_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hinge loss outside expanding ideal xy boxes."""
    if pred_xy.numel() == 0:
        return pred_xy.new_zeros(())
    B = pred_xy.size(0)
    pred = pred_xy.view(B, 3, 2)
    if module_mask is None:
        module_mask = torch.ones(B, 3, device=pred_xy.device, dtype=pred_xy.dtype)
    mask = module_mask.float()
    total = pred_xy.new_zeros(())
    count = pred_xy.new_zeros(())
    for i in range(B):
        state = canon_states[int(labels[i].item())]
        modules = ONTOLOGY_MODULE_SPECS.get(state, {})
        for module_name, spec in modules.items():
            idx = MODULE_ORDER.index(module_name)
            if mask[i, idx] < 0.5:
                continue
            b = boxes.box(spec["color"])
            x, y = pred[i, idx, 0], pred[i, idx, 1]
            dx = F.relu(b["x_lo"] - x) + F.relu(x - b["x_hi"])
            dy = F.relu(b["y_lo"] - y) + F.relu(y - b["y_hi"])
            total = total + dx + dy
            count = count + 1.0
    if count < 1:
        return pred_xy.new_zeros(())
    return total / count


# masks xy loss to ontology-on modules
def color_xy_module_mask_batch(
    labels: torch.Tensor,
    canon_states: Sequence[str],
) -> torch.Tensor:
    """Active-module mask (B, 3) from ontology state specs."""
    device = labels.device
    B = labels.size(0)
    masks = torch.zeros(B, 3, device=device, dtype=torch.float32)
    for i in range(B):
        state = canon_states[int(labels[i].item())]
        for module_name in ONTOLOGY_MODULE_SPECS.get(state, {}):
            masks[i, MODULE_ORDER.index(module_name)] = 1.0
    return masks


# cardinality loss (CE only, intensity_pred + cie_xy)


# soft on-probability from predicted intensity
def soft_on_prob_from_intensity(
    intensity: torch.Tensor,
    *,
    threshold: float = MODULE_ON_INTENSITY_THRESHOLD,
    temperature: float = DEFAULT_INTENSITY_ON_TEMPERATURE,
) -> torch.Tensor:
    return torch.sigmoid(
        (intensity - float(threshold)) / max(float(temperature), 1e-4)
    )


# intensity evidence for cardinality CE
def _intensity_mask_evidence(on_prob: torch.Tensor, state: str) -> torch.Tensor:
    target = on_prob.new_tensor(STATE_ON_MASK.get(state, (0.0, 0.0, 0.0)))
    return 1.0 - (on_prob - target).abs().mean()


# score for CIE xy inside NBS color region
def _cie_in_nbs_score(x: torch.Tensor, y: torch.Tensor, color: str) -> torch.Tensor:
    return torch.exp(-cie_xy_nbs_violation(x, y, color) * 10.0)


# color evidence from CIE xy heads
def _color_evidence_cie_xy(state: str, xy_pred: torch.Tensor, device: torch.device) -> torch.Tensor:
    if state == "off":
        return torch.ones((), device=device)
    modules = ONTOLOGY_MODULE_SPECS.get(state, {})
    if not modules:
        return torch.ones((), device=device)
    scores = []
    xy = xy_pred.view(3, 2)
    for module_name, spec in modules.items():
        idx = MODULE_IDX[module_name]
        scores.append(_cie_in_nbs_score(xy[idx, 0], xy[idx, 1], spec["color"]))
    if not scores:
        return torch.ones((), device=device)
    return torch.stack(scores).mean()


# cardinality CE over legal ontology states
def cardinality_consistency_loss(
    brightness_pred: torch.Tensor,
    labels: torch.Tensor,
    canon_states: Sequence[str],
    *,
    xy_pred: torch.Tensor | None = None,
    color_evidence_weight: float = 1.0,
    mode: str = "ce",
    intensity_pred: torch.Tensor | None = None,
    intensity_on_threshold: float = MODULE_ON_INTENSITY_THRESHOLD,
    intensity_temperature: float = DEFAULT_INTENSITY_ON_TEMPERATURE,
    states: Sequence[str] = CARDINALITY_STATES,
) -> torch.Tensor:
    """Cardinality CE over intensity on/off + CIE xy evidence (mode='ce' only)."""
    if mode != "ce":
        raise ValueError("4_2_simple supports cardinality mode='ce' only")
    if intensity_pred is None or intensity_pred.numel() == 0:
        zero = brightness_pred if brightness_pred.numel() else labels.new_zeros(())
        return zero.new_zeros(()) if hasattr(zero, "new_zeros") else torch.zeros(())

    if brightness_pred.numel() == 0:
        brightness_pred = intensity_pred.new_zeros(intensity_pred.shape)

    on_prob = soft_on_prob_from_intensity(
        intensity_pred,
        threshold=intensity_on_threshold,
        temperature=intensity_temperature,
    )
    B = intensity_pred.size(0)
    device = intensity_pred.device
    logits = intensity_pred.new_zeros((B, len(states)))
    eps = 1e-4

    for i in range(B):
        xy_i = xy_pred[i] if xy_pred is not None else None
        for j, state in enumerate(states):
            e_bright = _intensity_mask_evidence(on_prob[i], state)
            e_color = (
                _color_evidence_cie_xy(state, xy_i, device)
                if xy_i is not None
                else torch.ones((), device=device)
            )
            logits[i, j] = torch.log(e_bright.clamp_min(eps)) + (
                color_evidence_weight * torch.log(e_color.clamp_min(eps))
            )

    state_to_j = {s: j for j, s in enumerate(states)}
    target = labels.new_zeros((B,), dtype=torch.long)
    for i in range(B):
        name = canon_states[int(labels[i].item())]
        target[i] = int(state_to_j.get(name, state_to_j["off"]))
    return F.cross_entropy(logits, target)


# ontology text helpers (static maps, no RDF)


# maps a state name to ontology attribute values
def map_state_to_ontology_values(
    state: str, ontology_elements: list[str], conf: float
) -> dict[str, str]:
    if state not in ONTOLOGY_STATE_MAPPINGS:
        return {}
    values = {}
    for prop in ontology_elements:
        if prop in ONTOLOGY_STATE_MAPPINGS[state]:
            values[prop] = ONTOLOGY_STATE_MAPPINGS[state][prop]
    values["confidence"] = f"{conf:.3f}"
    return values


# builds a short text description for a state
def state_to_description(
    state: str,
    conf: float,
    ontology_elements: Optional[list[str]],
) -> str:
    if ontology_elements:
        attr_vals = map_state_to_ontology_values(state, ontology_elements, conf)
        attr_str = ", ".join(
            f"{k} is {v}" for k, v in attr_vals.items() if k != "confidence"
        )
        return (
            f"A photo of a traffic light showing the {state} state, "
            f"where the signal attributes are: {attr_str}."
        )
    return STATE_DESCRIPTIONS.get(state, f"The signal shows {state}.")


# dataset


# TrafficLightDataset class - crops + ontology / module targets
class TrafficLightDataset(Dataset):
    """
    Uniform return:
      (image, label, state, ontology_text, attr_targets,
       brightness_targets, brightness_valid,
       module_images_color, module_images_intensity)

    baseline: module crops empty; brightness zeros.
    card_owned: color thirds for xy head; gray thirds for intensity; brightness ratios.
    """

    def __init__(
        self,
        data_list,
        image_dir,
        transform=None,
        ontology_elements=None,
        use_brightness_targets: bool = False,
        use_module_crops: bool = False,
        module_crop_mode: str = "gray",
    ):
        self.data_list = data_list
        self.image_dir = image_dir
        self.transform = transform
        self.ontology_elements = ontology_elements or list(ATTRIBUTE_VALUE_MAPS.keys())
        self.use_brightness_targets = bool(use_brightness_targets)
        self.use_module_crops = bool(use_module_crops)
        self.module_crop_mode = str(module_crop_mode or "gray")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        row = self.data_list[idx]
        img_path = Path(self.image_dir) / row["filename"]
        image = Image.open(img_path).convert("RGB")

        state = row["state"]
        label = STATE_TO_IDX[state]

        if self.use_module_crops:
            module_pils = split_module_pils(image)

            def _stack_modules(pils):
                if self.transform:
                    return torch.stack([self.transform(mp) for mp in pils])
                return torch.stack([T.functional.to_tensor(mp) for mp in pils])

            module_images_color = _stack_modules(module_pils)
            if self.module_crop_mode == "gray":
                gray_pils = [pil_to_gray_rgb(mp) for mp in module_pils]
                module_images_intensity = _stack_modules(gray_pils)
            else:
                module_images_intensity = module_images_color
        else:
            module_images_color = torch.zeros(1, dtype=torch.float)
            module_images_intensity = torch.zeros(1, dtype=torch.float)

        if self.transform:
            image = self.transform(image)

        ontology_text = state_to_description(
            state, conf=1.0, ontology_elements=self.ontology_elements
        )
        attr_values = map_state_to_ontology_values(
            state, self.ontology_elements, conf=1.0
        )
        attr_targets = []
        for prop in self.ontology_elements:
            value = attr_values.get(prop, "N/A")
            target_idx = ATTRIBUTE_VALUE_MAPS.get(prop, {}).get(value, 0)
            attr_targets.append(target_idx)
        attr_targets = torch.tensor(attr_targets, dtype=torch.long)

        if self.use_brightness_targets:
            targets = row.get("brightness_targets", [0.0, 0.0, 0.0])
            brightness_targets = torch.tensor(targets, dtype=torch.float)
            brightness_valid = torch.tensor(
                1.0 if row.get("brightness_valid", False) else 0.0,
                dtype=torch.float,
            )
        else:
            brightness_targets = torch.zeros(3, dtype=torch.float)
            brightness_valid = torch.tensor(0.0, dtype=torch.float)

        return (
            image,
            label,
            state,
            ontology_text,
            attr_targets,
            brightness_targets,
            brightness_valid,
            module_images_color,
            module_images_intensity,
        )


# VLM (CLIP backbone + heads)


# VLM class - CLIP backbone with classifier and aux heads
class VLM(nn.Module):
    def __init__(self, clip_model, clip_pretrained, num_classes):
        super().__init__()
        self.backbone = "clip"
        self.num_classes = num_classes
        self.num_ontology_elements = len(ATTRIBUTE_VALUE_MAPS)

        if open_clip is None:
            raise ImportError("open_clip is required for CLIP backbone.")

        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model,
            pretrained=clip_pretrained,
            weights_only=False,
        )
        self.model = model
        self.preprocess = preprocess
        for param in self.model.parameters():
            param.requires_grad = True

        feature_dim = self.model.visual.output_dim
        self.classifier = nn.Linear(feature_dim, num_classes)

        total_attr_outputs = sum(len(v) for v in ATTRIBUTE_VALUE_MAPS.values())
        self.aux_head = nn.Linear(feature_dim, total_attr_outputs)
        self.tokenizer = open_clip.get_tokenizer(clip_model)

        # Brightness ranking (whole-image).
        self.regression_head = nn.Linear(feature_dim, 3)
        # Per-module CIE xy (shared Linear(D,2) on each third).
        self.xy_head = nn.Linear(feature_dim, 2)
        # Module intensity + state (gray thirds under card_owned).
        self.module_state_head = nn.Linear(feature_dim, NUM_MODULE_STATES)
        self.module_intensity_head = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.module_intensity_head.weight)
        nn.init.constant_(self.module_intensity_head.bias, 0.0)

    def forward(self, x, ontology_text_list=None):
        visual_features = self.model.encode_image(x).float()
        text_features = None
        if ontology_text_list is not None and self.tokenizer is not None:
            text_tokens = self.tokenizer(ontology_text_list).to(visual_features.device)
            text_features = self.model.encode_text(text_tokens).float()
        logits = self.classifier(visual_features)
        aux_logits = self.aux_head(visual_features)
        regression_output = self.regression_head(visual_features)
        return logits, aux_logits, regression_output, visual_features, text_features

    def encode_modules(self, module_images):
        """Encode (B, 3, C, H, W) thirds → (B, 3, D)."""
        B, M, C, H, W = module_images.shape
        flat = module_images.reshape(B * M, C, H, W)
        feats = self.model.encode_image(flat).float()
        return feats.reshape(B, M, -1)


# logging / metrics helpers


# sets up file and console logging
def setup_logging(save_dir, run_name):
    log_path = Path(save_dir) / "training.log"
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    logging.info(f"--- {run_name} ---")
    logging.info(f"Log file created at: {log_path.resolve()}")


# formats stage timing dict for logs
def _fmt_stage_timing(timing: dict[str, float], *, wall_s: float | None = None) -> str:
    items = [(k, float(v)) for k, v in timing.items() if float(v) > 0]
    if not items:
        return "(no timing)"
    total = float(wall_s) if wall_s and wall_s > 0 else sum(v for _, v in items)
    items.sort(key=lambda kv: kv[1], reverse=True)
    parts = []
    for k, v in items:
        pct = 100.0 * v / total if total > 0 else 0.0
        parts.append(f"{k}={v:.1f}s ({pct:.0f}%)")
    return " ".join(parts)


# one-vs-rest metrics from a confusion matrix
def _per_class_ovr_from_cm(cm: np.ndarray, label_names: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    total = float(cm.sum()) if cm.size else 0.0
    n = cm.shape[0]
    for i, name in enumerate(label_names):
        tp = float(cm[i, i]) if i < n else 0.0
        fp = float(cm[:, i].sum() - tp) if i < n else 0.0
        fn = float(cm[i, :].sum() - tp) if i < n else 0.0
        tn = float(total - tp - fp - fn)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        out[name] = {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "tpr": tpr,
            "fpr": fpr,
            "tnr": tnr,
            "fnr": fnr,
        }
    return out


# json default serializer for numpy / path types
def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


# train / validate


# runs one training epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, args):
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    timing = {"data_load": 0.0, "h2d": 0.0, "forward": 0.0, "backward": 0.0}
    epoch_t0 = time.perf_counter()

    mode = str(getattr(args, "mode", "card_owned") or "card_owned").strip().lower()
    card_owned = mode == "card_owned"

    AUX_LOSS_WEIGHT = float(getattr(args, "attribute_loss_weight", 0.19))
    CONTRASTIVE_LOSS_WEIGHT = float(getattr(args, "contrastive_loss_weight", 0.262))
    rank_weight = float(getattr(args, "brightness_rank_weight", 0.38))
    cardinality_weight = float(getattr(args, "cardinality_weight", 0.38))
    rank_margin = float(getattr(args, "brightness_rank_margin", 0.05))
    off_margin = float(getattr(args, "brightness_off_margin", 0.05))
    intensity_ranges = (
        getattr(args, "_module_intensity_ranges", None) or DEFAULT_MODULE_INTENSITY_RANGES
    )

    if not card_owned:
        AUX_LOSS_WEIGHT = 0.0
        CONTRASTIVE_LOSS_WEIGHT = 0.0

    attr_start_indices = {}
    current_index = 0
    attr_names = list(ATTRIBUTE_VALUE_MAPS.keys())
    for attr_name, val_map in ATTRIBUTE_VALUE_MAPS.items():
        attr_start_indices[attr_name] = current_index
        current_index += len(val_map)

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{getattr(args, 'epochs', '?')} Train",
        leave=True,
        dynamic_ncols=True,
        unit="batch",
    )

    use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (
            use_amp
            and getattr(args, "amp_dtype", "bf16") == "bf16"
            and torch.cuda.is_bf16_supported()
        )
        else torch.float16
    )
    scaler = getattr(args, "_grad_scaler", None)
    if use_amp and amp_dtype == torch.float16 and scaler is None:
        scaler = torch.cuda.amp.GradScaler()
        args._grad_scaler = scaler

    batch_iter = iter(progress_bar)
    while True:
        t_load0 = time.perf_counter()
        try:
            (
                images,
                labels,
                _,
                ontology_texts,
                attr_targets,
                brightness_targets,
                brightness_valid,
                module_images_color,
                module_images_intensity,
            ) = next(batch_iter)
        except StopIteration:
            break
        timing["data_load"] += time.perf_counter() - t_load0

        t_h2d0 = time.perf_counter()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if attr_targets is not None:
            attr_targets = attr_targets.to(device, non_blocking=True)
        if card_owned:
            brightness_targets = brightness_targets.to(device, non_blocking=True)
            brightness_valid = brightness_valid.to(device, non_blocking=True)
            module_images_color = module_images_color.to(device, non_blocking=True)
            module_images_intensity = module_images_intensity.to(
                device, non_blocking=True
            )
        timing["h2d"] += time.perf_counter() - t_h2d0

        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else contextlib.nullcontext()
        )

        t_fwd0 = time.perf_counter()
        with autocast_ctx:
            # baseline: CE only; card_owned still requests text for contrastive.
            text_in = ontology_texts if card_owned else None
            logits, aux_logits, regression_output, visual_features, text_features = model(
                images, text_in
            )

            primary_loss = criterion(logits, labels)
            loss = primary_loss

            attribute_loss = torch.tensor(0.0, device=device)
            contrastive_loss = torch.tensor(0.0, device=device)
            brightness_rank_loss = torch.tensor(0.0, device=device)
            color_xy_loss_val = torch.tensor(0.0, device=device)
            module_state_loss_val = torch.tensor(0.0, device=device)
            module_intensity_loss_val = torch.tensor(0.0, device=device)
            cardinality_loss_val = torch.tensor(0.0, device=device)

            if card_owned:
                # --- Attribute CE ---
                if AUX_LOSS_WEIGHT > 0 and aux_logits is not None:
                    for i, attr_name in enumerate(attr_names):
                        start_idx = attr_start_indices[attr_name]
                        num_values = len(ATTRIBUTE_VALUE_MAPS[attr_name])
                        attr_logits = aux_logits[:, start_idx : start_idx + num_values]
                        attribute_loss = attribute_loss + criterion(
                            attr_logits, attr_targets[:, i]
                        )
                    attribute_loss = attribute_loss / model.num_ontology_elements
                    loss = loss + AUX_LOSS_WEIGHT * attribute_loss

                # --- Brightness ranking (standalone under card_owned) ---
                brightness_rank_loss = module_brightness_ranking_loss(
                    regression_output,
                    labels,
                    CANON_STATES,
                    margin=rank_margin,
                    off_margin=off_margin,
                )
                loss = loss + rank_weight * brightness_rank_loss

                # --- Color thirds → CIE xy box hinge ---
                color_feats = model.encode_modules(module_images_color)
                color_xy_output = color_xy_from_module_feats(color_feats, model.xy_head)
                onto_m = color_xy_module_mask_batch(labels, CANON_STATES)
                xy_boxes = getattr(args, "_adaptive_xy_boxes", None)
                if xy_boxes is not None:
                    color_xy_loss_val = cie_xy_box_range_loss(
                        color_xy_output, labels, CANON_STATES, xy_boxes, onto_m
                    )
                    # expand boxes from model predictions + ontology module mask
                    xy_boxes.update_from_batch(
                        color_xy_output, labels, CANON_STATES, onto_m
                    )
                else:
                    color_xy_loss_val = torch.tensor(0.0, device=device)

                # --- Gray thirds → module state CE + softmax_rel intensity ---
                intensity_feats = model.encode_modules(module_images_intensity)
                state_logits = module_state_logits_from_feats(
                    intensity_feats, model.module_state_head
                )
                _, intensity_raw = module_intensity_from_feats(
                    intensity_feats,
                    model.module_intensity_head,
                    cd_map=getattr(args, "_module_cd_map", None),
                    return_logits=True,
                )
                state_targets = module_state_targets_from_labels(labels, CANON_STATES)
                module_state_loss_val = module_state_ce_loss(state_logits, state_targets)
                module_intensity_loss_val, intensity_pred = module_softmax_rel_loss(
                    intensity_raw,
                    state_targets=state_targets,
                    brightness_r=brightness_targets,
                    brightness_valid=brightness_valid,
                    cd_map=getattr(args, "_module_cd_map", None),
                    ranges=intensity_ranges,
                )

                # --- Cardinality CE; fold state/intensity/xy into L_Card ---
                # WHY fold: card_owned recipe treats module/color as evidence
                # inside one cardinality scalar, scaled once by cardinality_weight
                # (not as separate weighted terms).
                card_brightness = (
                    regression_output
                    if regression_output is not None
                    else intensity_pred.new_zeros(intensity_pred.shape)
                )
                cardinality_loss_val = cardinality_consistency_loss(
                    card_brightness,
                    labels,
                    CANON_STATES,
                    xy_pred=color_xy_output,
                    color_evidence_weight=float(
                        getattr(args, "cardinality_color_weight", 1.0)
                    ),
                    mode="ce",
                    intensity_pred=intensity_pred,
                    intensity_on_threshold=MODULE_ON_INTENSITY_THRESHOLD,
                    intensity_temperature=float(
                        getattr(args, "cardinality_intensity_temperature", 10.0)
                    ),
                )
                cardinality_loss_val = (
                    cardinality_loss_val
                    + module_state_loss_val
                    + module_intensity_loss_val
                    + color_xy_loss_val
                )
                loss = loss + cardinality_weight * cardinality_loss_val

                # --- Contrastive ---
                if (
                    CONTRASTIVE_LOSS_WEIGHT > 0
                    and visual_features is not None
                    and text_features is not None
                ):
                    visual_features = visual_features / visual_features.norm(
                        dim=-1, keepdim=True
                    )
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )
                    contrastive_loss = torch.mean(
                        1.0 - torch.sum(visual_features * text_features, dim=-1)
                    )
                    loss = loss + CONTRASTIVE_LOSS_WEIGHT * contrastive_loss

        timing["forward"] += time.perf_counter() - t_fwd0

        t_bwd0 = time.perf_counter()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        timing["backward"] += time.perf_counter() - t_bwd0

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        postfix = {
            "L_Comb": f"{loss.item():.4f}",
            "L_P": f"{primary_loss.item():.4f}",
        }
        if card_owned:
            postfix.update(
                {
                    "L_A": f"{attribute_loss.item():.4f}",
                    "L_C": f"{contrastive_loss.item():.4f}",
                    "L_Br": f"{brightness_rank_loss.item():.4f}",
                    "L_Xy": f"{color_xy_loss_val.item():.4f}",
                    "L_Card": f"{cardinality_loss_val.item():.4f}",
                    "L_MSt": f"{module_state_loss_val.item():.4f}",
                    "L_MInt": f"{module_intensity_loss_val.item():.4f}",
                }
            )
        progress_bar.set_postfix(postfix)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    train_wall = time.perf_counter() - epoch_t0
    timing["other"] = max(0.0, train_wall - sum(timing.values()))
    logging.info(
        f"Epoch {epoch} Train: Loss (Combined)={avg_loss:.4f}, Accuracy={avg_acc:.4f}"
    )
    logging.info(
        f"Epoch {epoch} Train timing ({train_wall:.1f}s): "
        f"{_fmt_stage_timing(timing, wall_s=train_wall)}"
    )
    return avg_loss, avg_acc, timing, train_wall


# runs validation and returns metrics
def validate(model, dataloader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_states = []
    timing = {"data_load": 0.0, "h2d": 0.0, "forward": 0.0}
    val_t0 = time.perf_counter()

    with torch.no_grad():
        use_amp = bool(getattr(args, "amp", False)) and device.type == "cuda"
        amp_dtype = (
            torch.bfloat16
            if (
                use_amp
                and getattr(args, "amp_dtype", "bf16") == "bf16"
                and torch.cuda.is_bf16_supported()
            )
            else torch.float16
        )
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        data_iter = tqdm(
            dataloader, desc="Validate", leave=True, dynamic_ncols=True, unit="batch"
        )
        for batch in data_iter:
            t_load0 = time.perf_counter()
            images, labels, states = batch[0], batch[1], batch[2]
            timing["data_load"] += time.perf_counter() - t_load0

            t_h2d0 = time.perf_counter()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            timing["h2d"] += time.perf_counter() - t_h2d0

            t_fwd0 = time.perf_counter()
            with autocast_ctx:
                logits, _, _, _, _ = model(images, ontology_text_list=None)
                loss = criterion(logits, labels)
            timing["forward"] += time.perf_counter() - t_fwd0
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_states.extend(states)

    avg_loss = total_loss / max(len(dataloader.dataset), 1)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    val_wall = time.perf_counter() - val_t0
    timing["other"] = max(0.0, val_wall - sum(timing.values()))
    logging.info(
        f"Epoch Validate: Loss={avg_loss:.4f}, Accuracy={acc:.4f}, Macro F1={f1_macro:.4f}"
    )
    logging.info(
        f"Epoch Validate timing ({val_wall:.1f}s): {_fmt_stage_timing(timing, wall_s=val_wall)}"
    )

    label_ids = list(range(len(CANON_STATES)))
    label_names = list(CANON_STATES)
    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    report = classification_report(
        all_labels,
        all_preds,
        labels=label_ids,
        target_names=label_names,
        output_dict=False,
        zero_division=0,
    )
    logging.info("\n--- Detailed Classification Report ---")
    logging.info(f"\n{report}")
    cm = confusion_matrix(all_labels, all_preds, labels=label_ids)
    logging.info("\n--- Confusion Matrix (True Rows, Pred Columns) ---")
    logging.info(f"States (Columns/Predicted): {label_names}")
    logging.info(f"\n{cm}")
    logging.info("----------------------------------\n")

    per_class = _per_class_ovr_from_cm(cm, label_names)
    metrics = {
        "val_loss": float(avg_loss),
        "val_f1": float(f1_macro),
        "val_accuracy": float(acc),
        "confusion_matrix": {"labels": label_names, "matrix": cm.tolist()},
        "per_class": per_class,
        "classification_report": report_dict,
    }
    return avg_loss, f1_macro, timing, val_wall, metrics


# run training


# end-to-end training loop for baseline or card_owned
def run_training(args):
    setup_logging(args.save_dir, f"{args.backbone}_{args.mode}_run")
    logging.info(f"Configuration: {json.dumps(vars(args), indent=2, default=str)}")

    device = torch.device(args.device)
    device_info = f"Using device: {device}"
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            device_info += f" ({torch.cuda.get_device_name(device)})"
        except Exception:
            device_info += " (GPU detected)"
    logging.info(device_info)

    mode = str(args.mode).strip().lower()
    card_owned = mode == "card_owned"
    if mode not in ("baseline", "card_owned"):
        logging.error(f"Unknown --mode={mode}; choose baseline|card_owned")
        return

    # force recipe flags for the selected mode (pipeline leftover flags ignored)
    if card_owned:
        args.module_crop_mode = "gray"
        args.color_method = "cie_xy_ideal_expand"
        args.xy_expand_source = "pred"
        args.module_onoff_method = "softmax_rel"
        args.cardinality_mode = "ce"
        args.brightness_rank_loss = True
        args.cardinality_loss = True
        args.module_intensity_state_loss = True
        logging.info(
            "card_owned recipe: gray intensity crops | cie_xy_ideal_expand "
            "(xy_expand_source=pred) | softmax_rel | brightness_rank ON | "
            "fold state+intensity+xy into L_Card"
        )
    else:
        args.attribute_loss_weight = 0.0
        args.contrastive_loss_weight = 0.0
        args.brightness_rank_weight = 0.0
        args.cardinality_weight = 0.0
        args.module_state_weight = 0.0
        args.module_intensity_weight = 0.0
        args.color_method_weight = 0.0
        args.brightness_rank_loss = False
        args.cardinality_loss = False
        args.module_intensity_state_loss = False
        logging.info("baseline → classification CE only (all aux weights forced to 0)")

    required_attrs = load_attr_filters(args.attr_filter_json)
    logging.info(f"DTLD attribute filter: {required_attrs}")
    logging.info(f"DTLD labels dir: {args.labels_dir}")

    all_data, filter_stats = scan_filtered_crop_records(
        image_dir=args.image_dir,
        labels_dir=args.labels_dir,
        required_attrs=required_attrs,
        exclude_unknown=not args.include_unknown,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    filter_manifest_path = Path(args.save_dir) / "filtered_dataset.json"
    with open(filter_manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_dir": args.image_dir,
                "labels_dir": args.labels_dir,
                "required_attributes": required_attrs,
                "exclude_unknown": not args.include_unknown,
                "stats": filter_stats,
                "n_records": len(all_data),
                "state_counts": {
                    s: int(c)
                    for s, c in pd.Series([r["state"] for r in all_data])
                    .value_counts()
                    .items()
                },
            },
            f,
            indent=2,
        )
    logging.info(f"Filter stats: {json.dumps(filter_stats, indent=2)}")

    if not all_data:
        logging.error(
            "No crops passed the DTLD attribute filter. "
            "Check --image_dir, --labels_dir, and --attr_filter_json."
        )
        return

    max_per_class = getattr(args, "max_per_class", None)
    if max_per_class is not None and int(max_per_class) > 0:
        max_per_class = int(max_per_class)
        rng = random.Random(int(getattr(args, "seed", 42)))
        by_state: dict[str, list] = defaultdict(list)
        for rec in all_data:
            by_state[str(rec.get("state", ""))].append(rec)
        sampled = []
        for state in CANON_STATES:
            pool = by_state.get(state, [])
            if len(pool) > max_per_class:
                pool = rng.sample(pool, max_per_class)
            sampled.extend(pool)
        for state, pool in by_state.items():
            if state in CANON_STATES:
                continue
            if len(pool) > max_per_class:
                pool = rng.sample(pool, max_per_class)
            sampled.extend(pool)
        before = len(all_data)
        all_data = sampled
        logging.info(
            f"--max_per_class={max_per_class}: subsampled {before} → {len(all_data)} "
            f"(seed={getattr(args, 'seed', 42)})"
        )

    df = pd.DataFrame(all_data)
    logging.info(f"Accepted {len(df)} filtered crops. States: {sorted(df['state'].unique())}")
    logging.info(f"Class Counts:\n{df['state'].value_counts()}")

    args._module_cd_map = None
    args._adaptive_xy_boxes = None
    args._module_intensity_ranges = dict(DEFAULT_MODULE_INTENSITY_RANGES)

    if card_owned:
        if not getattr(args, "dtld_root", None):
            logging.error(
                "card_owned requires --dtld_root for zone/ring brightness targets "
                "(softmax_rel)."
            )
            return
        cache_path = getattr(args, "brightness_cache", None)
        if not cache_path:
            cache_path = str(
                Path(args.save_dir)
                / f"brightness_targets_border{args.brightness_ring_border}.json"
            )
        logging.info(
            "Precomputing zone/ring brightness ratios "
            f"(ring border={args.brightness_ring_border})..."
        )
        brightness_stats = enrich_records_with_zone_ring_ratios(
            all_data,
            image_dir=args.image_dir,
            dtld_root=args.dtld_root,
            border=args.brightness_ring_border,
            cache_path=cache_path,
            fast_tiff=True,
        )
        logging.info(f"Brightness target stats: {json.dumps(brightness_stats, indent=2)}")
        logging.info(f"Brightness cache: {cache_path}")
        df = pd.DataFrame(all_data)

    if len(df["state"].unique()) > 1 and len(df) > 1:
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=args.seed, stratify=df["state"]
        )
    else:
        logging.warning(
            "Insufficient data (1 class or < 2 samples). Skipping stratified split."
        )
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=args.seed)

    if card_owned:
        cd_map = fit_cd_map_from_records(train_df.to_dict("records"), CANON_STATES)
        args._module_cd_map = cd_map
        logging.info(f"Fitted module r→cd map: {cd_map.to_dict()}")
        args._adaptive_xy_boxes = AdaptiveXyBoxes(ideal_xy=IDEAL_XY)
        logging.info(
            "CIE xy ideal-expand boxes (expand from model predictions): "
            + " | ".join(
                f"{c}=({IDEAL_XY[c][0]:.3f},{IDEAL_XY[c][1]:.3f})"
                for c in TRAFFIC_COLORS
            )
        )

    if str(getattr(args, "backbone", "clip")).lower() != "clip":
        logging.warning(
            f"--backbone={args.backbone!r} ignored; 4_2_simple uses CLIP only."
        )
    model = VLM(args.clip_model, args.clip_pretrained, NUM_CLASSES).to(device)
    transform = model.preprocess

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if bool(getattr(args, "amp", False)):
            logging.info(f"AMP enabled (dtype={getattr(args, 'amp_dtype', 'bf16')})")

    ontology_elements = list(ATTRIBUTE_VALUE_MAPS.keys())
    train_dataset = TrafficLightDataset(
        train_df.to_dict("records"),
        args.image_dir,
        transform=transform,
        ontology_elements=ontology_elements,
        use_brightness_targets=card_owned,
        use_module_crops=card_owned,
        module_crop_mode="gray" if card_owned else "color",
    )
    # Val: classification only — skip module crops / brightness.
    val_dataset = TrafficLightDataset(
        val_df.to_dict("records"),
        args.image_dir,
        transform=transform,
        ontology_elements=ontology_elements,
        use_brightness_targets=False,
        use_module_crops=False,
    )

    class_counts = train_df["state"].value_counts().reindex(CANON_STATES).fillna(0).to_dict()
    num_samples = len(train_df)
    weights_map = {
        state: num_samples / (NUM_CLASSES * count) if count > 0 else 1.0
        for state, count in class_counts.items()
    }
    train_weights = [weights_map[state] for state in train_df["state"]]
    sampler = WeightedRandomSampler(
        weights=train_weights, num_samples=num_samples, replacement=True
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 4))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )

    criterion = nn.CrossEntropyLoss()
    logging.info("Using standard CrossEntropyLoss with WeightedRandomSampler.")
    logging.info(
        "Loss weights: "
        f"attribute={args.attribute_loss_weight}, "
        f"contrastive={args.contrastive_loss_weight}, "
        f"brightness_rank={args.brightness_rank_weight}, "
        f"cardinality={args.cardinality_weight} "
        f"(folds state/intensity/xy under card_owned)"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    checkpoint_path = Path(args.save_dir) / "best_model.pt"
    metadata_path = Path(args.save_dir) / "best_model.json"

    def _fmt_duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        return f"{m}m{s:02d}s"

    train_start = time.perf_counter()
    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        logging.info(f"===== Epoch {epoch}/{args.epochs} =====")
        _, _, train_timing, train_wall = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args
        )
        val_loss, val_f1, val_timing, val_wall, val_metrics = validate(
            model, val_loader, criterion, device, args
        )
        scheduler.step()

        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_times.append(epoch_elapsed)
        remaining = args.epochs - epoch
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta_s = avg_epoch * remaining
        total_elapsed = time.perf_counter() - train_start
        logging.info(
            f"Epoch {epoch}/{args.epochs} done in {_fmt_duration(epoch_elapsed)} | "
            f"elapsed {_fmt_duration(total_elapsed)} | "
            f"ETA {_fmt_duration(eta_s)} ({remaining} epoch(s) left) | "
            f"val_f1={val_f1:.4f} best_f1={max(best_f1, val_f1):.4f}"
        )
        logging.info(
            f"Epoch {epoch} stage timing: "
            f"{_fmt_stage_timing({'train': train_wall, 'validate': val_wall}, wall_s=epoch_elapsed)}"
        )
        xy_boxes = getattr(args, "_adaptive_xy_boxes", None)
        if xy_boxes is not None:
            logging.info(
                "Adaptive xy boxes: "
                + " | ".join(
                    f"{c}=x[{b['x_lo']:.3f},{b['x_hi']:.3f}] "
                    f"y[{b['y_lo']:.3f},{b['y_hi']:.3f}]"
                    for c, b in xy_boxes.snapshot().items()
                )
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            meta = {
                "epoch": epoch,
                "val_f1": best_f1,
                "val_loss": float(val_loss),
                "val_accuracy": float(val_metrics.get("val_accuracy") or 0.0),
                "confusion_matrix": val_metrics.get("confusion_matrix"),
                "per_class": val_metrics.get("per_class"),
                "classification_report": json.loads(
                    json.dumps(
                        val_metrics.get("classification_report"), default=_json_default
                    )
                ),
                "mode": mode,
                "loss_layout": mode,
                "brightness_rank_loss": bool(card_owned),
                "color_method": "cie_xy_ideal_expand" if card_owned else None,
                "xy_expand_source": "pred" if card_owned else None,
                "band_expand_source": "predicted_xy" if card_owned else None,
                "cardinality_loss": bool(card_owned),
                "cardinality_mode": "ce",
                "cardinality_on_source": "intensity" if card_owned else None,
                "module_intensity_state_loss": bool(card_owned),
                "module_crop_mode": "gray" if card_owned else "na",
                "module_onoff_method": "softmax_rel" if card_owned else None,
                "module_cd_map": (
                    args._module_cd_map.to_dict()
                    if getattr(args, "_module_cd_map", None) is not None
                    else None
                ),
                "module_on_intensity_threshold": MODULE_ON_INTENSITY_THRESHOLD,
                "module_intensity_ranges": args._module_intensity_ranges,
                "attribute_loss_weight": float(args.attribute_loss_weight),
                "contrastive_loss_weight": float(args.contrastive_loss_weight),
                "brightness_rank_weight": float(args.brightness_rank_weight),
                "cardinality_weight": float(args.cardinality_weight),
                "module_state_weight": float(args.module_state_weight),
                "module_intensity_weight": float(args.module_intensity_weight),
                "color_method_weight": float(args.color_method_weight),
                "ideal_xy": {c: list(IDEAL_XY[c]) for c in IDEAL_XY},
                "color_from_modules": True,
                "color_head_layout": "shared_per_third",
            }
            if xy_boxes is not None:
                meta["adaptive_xy_boxes"] = xy_boxes.snapshot()
            with open(metadata_path, "w") as f:
                json.dump(meta, f, indent=2)
            logging.info(
                f"Saved new best model (epoch {epoch}, F1={best_f1:.4f}) to {checkpoint_path}"
            )

    logging.info(
        f"Training finished in {_fmt_duration(time.perf_counter() - train_start)} "
        f"(best val F1={best_f1:.4f})"
    )


# CLI


# parses CLI arguments
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Simplified two-mode VLM trainer: baseline (CE only) or "
            "card_owned (sweep_full3 combo-1 ontology recipe)."
        )
    )
    p.add_argument(
        "--mode",
        type=str,
        default="card_owned",
        choices=["baseline", "card_owned"],
        help="Training mode (default: card_owned).",
    )
    p.add_argument(
        "--loss_layout",
        type=str,
        default=None,
        choices=["baseline", "card_owned"],
        help="Alias for --mode (pipeline drop-in).",
    )

    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--labels_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--dtld_root", type=str, default=None)
    p.add_argument("--backbone", type=str, default="clip")
    p.add_argument("--clip_model", type=str, default="ViT-B-32")
    p.add_argument("--clip_pretrained", type=str, default="openai")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--attr_filter_json", type=str, default=None)
    p.add_argument("--include_unknown", action="store_true")
    p.add_argument("--brightness_cache", type=str, default=None)
    p.add_argument("--brightness_ring_border", type=int, default=20)

    # Weights (card_owned defaults = sweep_full3 combo 1).
    p.add_argument(
        "--attribute_loss_weight",
        type=float,
        default=DEFAULT_CARD_OWNED_WEIGHTS["attribute"],
    )
    p.add_argument(
        "--contrastive_loss_weight",
        type=float,
        default=DEFAULT_CARD_OWNED_WEIGHTS["contrastive"],
    )
    p.add_argument(
        "--brightness_rank_weight",
        type=float,
        default=DEFAULT_CARD_OWNED_WEIGHTS["brightness_rank"],
    )
    p.add_argument(
        "--cardinality_weight",
        type=float,
        default=DEFAULT_CARD_OWNED_WEIGHTS["cardinality"],
    )
    p.add_argument(
        "--module_state_weight",
        type=float,
        default=DEFAULT_CARD_OWNED_WEIGHTS["module_state"],
        help="Parsed for CLI compat; folded into cardinality under card_owned.",
    )
    p.add_argument(
        "--module_intensity_weight",
        type=float,
        default=DEFAULT_CARD_OWNED_WEIGHTS["module_intensity"],
        help="Parsed for CLI compat; folded into cardinality under card_owned.",
    )
    p.add_argument(
        "--color_method_weight",
        type=float,
        default=DEFAULT_CARD_OWNED_WEIGHTS["color_method"],
        help="Accepted for CLI compat; unused standalone under card_owned fold.",
    )
    p.add_argument("--brightness_rank_margin", type=float, default=0.05)
    p.add_argument("--brightness_off_margin", type=float, default=0.05)
    p.add_argument("--cardinality_color_weight", type=float, default=1.0)
    p.add_argument("--cardinality_intensity_temperature", type=float, default=10.0)

    # accept-and-ignore leftover pipeline flags (forced by recipe when mode set)
    p.add_argument("--color_method", type=str, default=None)
    p.add_argument("--xy_expand_source", type=str, default=None)
    p.add_argument("--color_hue_weight", type=float, default=0.0)
    p.add_argument("--color_xy_weight", type=float, default=0.0)
    p.add_argument("--hue_cache", type=str, default=None)
    p.add_argument("--hue_measure_mode", type=str, default=None)
    p.add_argument("--cie_xy_cache", type=str, default=None)
    p.add_argument("--module_intensity_state_loss", action="store_true")
    p.add_argument("--module_onoff_method", type=str, default=None)
    p.add_argument("--module_crop_mode", type=str, default=None)
    p.add_argument("--cardinality_loss", action="store_true")
    p.add_argument("--cardinality_mode", type=str, default=None)
    p.add_argument("--brightness_rank_loss", action="store_true")

    return p.parse_args()


# entry point
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --loss_layout aliases --mode for pipeline drop-in.
    if args.loss_layout is not None:
        args.mode = args.loss_layout

    mode = str(args.mode).strip().lower()
    if mode == "baseline":
        args.attribute_loss_weight = 0.0
        args.contrastive_loss_weight = 0.0
        args.brightness_rank_weight = 0.0
        args.cardinality_weight = 0.0
        args.module_state_weight = 0.0
        args.module_intensity_weight = 0.0
        args.color_method_weight = 0.0

    logging.info(f"Starting training (mode={mode})...")
    run_training(args)


if __name__ == "__main__":
    main()
