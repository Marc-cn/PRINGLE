#!/usr/bin/env python3
"""
cardinality consistency evaluator for clean/baseline + ontology checkpoints.

Metrics:
  - card_ok         evidence matches a legal ontology state
  - catch_rate      UNCERTAIN | pred != gt
  - false_alarm_rate UNCERTAIN | pred == gt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
except ImportError as e:  # pragma: no cover
    raise ImportError("open_clip is required") from e

from helper import load_attr_filters, scan_filtered_crop_records, DEFAULT_CONS_OUT, DEFAULT_CROPS_DIR, DEFAULT_LABELS_DIR
# shared ontology / module helpers
from ontology_common import (
    CANON_STATES,
    IDEAL_XY,
    MODULE_COLORS,
    MODULE_ON_INTENSITY_THRESHOLD,
    MODULE_ORDER,
    MODULE_SLOT_COLOR,
    NUM_MODULES,
    NUM_MODULE_STATES,
    ONTOLOGY_MODULE_SPECS,
    STATE_ON_MASK,
    CdMap,
    color_xy_from_module_feats,
    infer_on_mask_from_intensity,
    module_intensity_from_feats,
    pil_to_gray_rgb,
    split_module_pils,
    split_three_zones,
)

DEFAULT_IMAGE_DIR = DEFAULT_CROPS_DIR
DEFAULT_OUT_DIR = DEFAULT_CONS_OUT


# NBS Handbook 95 Table I-1 general signal-color masks (CIE xy)

def mask_signal_red(X: Any, Y: Any) -> Any:
    return ((X + Y) >= 0.992) & ((X - Y) >= 0.330)


def mask_signal_yellow(X: Any, Y: Any) -> Any:
    return (Y >= 0.383) & ((X + Y) >= (0.872 + 0.200 * X)) & (X >= 0.550)


def mask_signal_green(X: Any, Y: Any) -> Any:
    return (X <= (0.360 - 0.080 * Y)) & (X <= (0.650 * Y)) & (Y >= (0.390 - 0.171 * X))


NBS_SIGNAL_MASKS = {
    "red": mask_signal_red,
    "yellow": mask_signal_yellow,
    "green": mask_signal_green,
}


def _xy_inside_box(x: float, y: float, box: dict[str, float]) -> bool:
    return box["x_lo"] <= x <= box["x_hi"] and box["y_lo"] <= y <= box["y_hi"]


# label module color and CIE xy coordinates
def label_module_color_xy(
    x: float,
    y: float,
    boxes: dict[str, dict[str, float]] | None,
    *,
    prefer_box: bool = True,
) -> tuple[str, bool]:
    """Return (color_or_other, in_nbs_for_that_color)."""
    in_nbs = {
        c: bool(NBS_SIGNAL_MASKS[c](np.asarray(x), np.asarray(y)))
        for c in MODULE_COLORS
    }
    if prefer_box and boxes:
        hits = []
        for color in MODULE_COLORS:
            if color in boxes and _xy_inside_box(x, y, boxes[color]):
                box = boxes[color]
                cx = 0.5 * (box["x_lo"] + box["x_hi"])
                cy = 0.5 * (box["y_lo"] + box["y_hi"])
                hits.append((color, (x - cx) ** 2 + (y - cy) ** 2))
        if hits:
            hits.sort(key=lambda z: z[1])
            c = hits[0][0]
            return c, in_nbs[c]
        return "other", any(in_nbs.values())
    nbs_hits = [c for c, ok in in_nbs.items() if ok]
    if len(nbs_hits) == 1:
        return nbs_hits[0], True
    if not nbs_hits:
        return "other", False
    return "other", True


# matches on mask to a legal ontology state
def evidence_state_from_modules(
    on_mask: Sequence[bool] | np.ndarray,
    module_colors: dict[str, str],
) -> str | None:
    """Match on/off + ON-module colors to a legal ontology state, else None (UNCERTAIN)."""
    observed_on = tuple(bool(on_mask[i]) for i in range(3))
    for state, expected_mask in STATE_ON_MASK.items():
        expected_on = tuple(m > 0.5 for m in expected_mask)
        if observed_on != expected_on:
            continue
        if state == "off":
            return "off"
        specs = ONTOLOGY_MODULE_SPECS.get(state, {})
        if not specs:
            continue
        if all(
            module_colors.get(name, "other") == spec["color"]
            for name, spec in specs.items()
        ):
            return state
    return None

# checks if on mask is a legal traffic-light pattern
def on_mask_legal(on_mask: Sequence[bool] | np.ndarray) -> bool:
    observed = tuple(bool(on_mask[i]) for i in range(3))
    for expected_mask in STATE_ON_MASK.values():
        if observed == tuple(m > 0.5 for m in expected_mask):
            return True
    return False

# checks if module colors are consistent with the ontology state
def color_ok_independent(
    on_mask: Sequence[bool] | np.ndarray,
    module_colors: dict[str, str],
) -> bool:
    for i, name in enumerate(MODULE_ORDER):
        if not bool(on_mask[i]):
            continue
        if module_colors.get(name, "other") != MODULE_SLOT_COLOR[name]:
            return False
    return True

# formats consistency explanation into a human-readable string
def format_consistency_explanation(
    *,
    on_mask: Sequence[bool] | np.ndarray,
    module_colors: dict[str, str],
    pred_state: str,
    evidence_state: str | None,
    card_ok: bool,
    on_ok: bool,
    color_ok: bool,
    pred_consistent: bool,
    module_intensities: dict[str, float] | None = None,
    module_xy: dict[str, Sequence[float] | None] | None = None,
    intensity_threshold: float | None = None,
) -> str:
    """Consistency checker output."""
    active = [MODULE_ORDER[i] for i in range(3) if bool(on_mask[i])]
    active_str = ", ".join(active) if active else "none"
    color_bits = [
        f"{MODULE_ORDER[i]}={module_colors.get(MODULE_ORDER[i], 'other')}"
        for i in range(3)
        if bool(on_mask[i])
    ]
    colors_str = ", ".join(color_bits) if color_bits else "n/a"
    pred = str(pred_state or "").strip() or "unknown"

    evidence_bits: list[str] = []
    if module_intensities:
        thr = (
            f" (on if I≥{float(intensity_threshold):.0f})"
            if intensity_threshold is not None
            else ""
        )
        i_parts = []
        for name in MODULE_ORDER:
            if name not in module_intensities:
                continue
            val = float(module_intensities[name])
            on_tag = "on" if bool(on_mask[MODULE_ORDER.index(name)]) else "off"
            i_parts.append(f"{name}={val:.1f}[{on_tag}]")
        if i_parts:
            evidence_bits.append(f"Intensities{thr}: {', '.join(i_parts)}")
    if module_xy:
        xy_parts = []
        for name in MODULE_ORDER:
            xy = module_xy.get(name)
            if xy is None or len(xy) < 2:
                continue
            try:
                x, y = float(xy[0]), float(xy[1])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            lab = module_colors.get(name, "other")
            xy_parts.append(f"{name}=({x:.3f},{y:.3f})→{lab}")
        if xy_parts:
            evidence_bits.append(f"CIE xy: {', '.join(xy_parts)}")
    evidence_suffix = (" " + " ".join(evidence_bits)) if evidence_bits else ""

    if card_ok:
        cons = "consistent" if pred_consistent else f"differs from prediction ({pred})"
        return (
            f"VALID: Evidence matches ontology state '{evidence_state}'. "
            f"Active modules: {active_str}. "
            f"Colors: {colors_str}. "
            f"Predicted state: {pred} ({cons})."
            f"{evidence_suffix}"
        )

    reasons: list[str] = []
    if not on_ok:
        reasons.append(
            f"on-mask {{{active_str}}} is not a legal traffic-light pattern"
        )
    if not color_ok:
        for i, name in enumerate(MODULE_ORDER):
            if not bool(on_mask[i]):
                continue
            got = module_colors.get(name, "other")
            expected = MODULE_SLOT_COLOR[name]
            if got != expected:
                reasons.append(
                    f"{name} is {got} (expected {expected} for that slot)"
                )
    if not reasons:
        reasons.append("configuration does not match any legal ontology state")

    return (
        f"UNCERTAIN: Cardinality of predicted state ({pred}) is violated. "
        f"Active modules: {active_str}. "
        f"Colors: {colors_str}. "
        f"Reasons: {'; '.join(reasons)}."
        f"{evidence_suffix}"
    )

# applies evidence metrics to a single record
def apply_evidence_metrics(
    *,
    on_mask: Sequence[bool] | np.ndarray,
    module_colors: dict[str, str],
    pred_state: str,
    module_intensities: dict[str, float] | None = None,
    module_xy: dict[str, Sequence[float] | None] | None = None,
    intensity_threshold: float | None = None,
) -> dict[str, Any]:
    on_ok = on_mask_legal(on_mask)
    color_ok = color_ok_independent(on_mask, module_colors)
    card_ok = bool(on_ok and color_ok)
    evidence_state = evidence_state_from_modules(on_mask, module_colors)
    pred_consistent = bool(card_ok and evidence_state == pred_state)
    explanation = format_consistency_explanation(
        on_mask=on_mask,
        module_colors=module_colors,
        pred_state=pred_state,
        evidence_state=evidence_state,
        card_ok=card_ok,
        on_ok=on_ok,
        color_ok=color_ok,
        pred_consistent=pred_consistent,
        module_intensities=module_intensities,
        module_xy=module_xy,
        intensity_threshold=intensity_threshold,
    )
    return {
        "evidence_state": evidence_state,
        "card_ok": card_ok,
        "pred_consistent": pred_consistent,
        "on_ok": on_ok,
        "color_ok": color_ok,
        "invalid": not card_ok,
        "explanation": explanation,
    }


# CardinalityHeads class - CLIP + heads needed for card_owned
class CardinalityHeads(nn.Module):
    """CLIP + heads needed for card_owned (classifier, xy, intensity, state)."""

    def __init__(self, clip_model: str = "ViT-B-32", num_classes: int = 5):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model, pretrained=None
        )
        self.model = model
        self.preprocess = preprocess
        feat = model.visual.output_dim
        self.classifier = nn.Linear(feat, num_classes)
        self.regression_head = nn.Linear(feat, 3)
        self.xy_head = nn.Linear(feat, 2)
        self.module_state_head = nn.Linear(feat, NUM_MODULE_STATES)
        self.module_intensity_head = nn.Linear(feat, 1)

    def forward(self, x: torch.Tensor):
        feats = self.model.encode_image(x).float()
        return self.classifier(feats), self.regression_head(feats), feats

    def encode_modules(self, module_images: torch.Tensor) -> torch.Tensor:
        B, M, C, H, W = module_images.shape
        flat = module_images.reshape(B * M, C, H, W)
        feats = self.model.encode_image(flat).float()
        return feats.reshape(B, M, -1)

# loads a cardinality checkpoint and prepares it for evaluation
def load_model(ckpt_path: Path, device: torch.device) -> CardinalityHeads:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    num_classes = 5
    for k, v in state.items():
        if k.endswith("classifier.weight"):
            num_classes = int(v.shape[0])
            break
    model = CardinalityHeads(num_classes=num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if "classifier.weight" not in state:
        raise KeyError(f"{ckpt_path} missing classifier.weight")
    model.has_module_intensity_head = "module_intensity_head.weight" in state
    model.has_module_state_head = "module_state_head.weight" in state
    model.has_xy_head = "xy_head.weight" in state
    if missing:
        print(f"load_state_dict missing (ok if baseline): {sorted(missing)[:8]}...")
    if unexpected:
        print(f"load_state_dict unexpected (ignored): {sorted(unexpected)[:8]}...")
    model.to(device).eval()
    return model


# stratified sampling of records by state

def stratified_sample(
    records: list[dict[str, Any]],
    max_per_state: int | None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if max_per_state is None or max_per_state <= 0:
        return records
    rng = random.Random(seed)
    by_state: dict[str, list] = defaultdict(list)
    for r in records:
        by_state[str(r.get("state", ""))].append(r)
    out = []
    for state, rows in sorted(by_state.items()):
        if len(rows) <= max_per_state:
            out.extend(rows)
        else:
            out.extend(rng.sample(rows, max_per_state))
    return out

# loads benign crops from a dataset root
def load_benign_records(
    image_dir: str,
    labels_dir: str,
    *,
    max_per_state: int | None,
    full: bool,
    seed: int,
    attr_filter_json: str | None = None,
) -> list[dict[str, Any]]:
    required = load_attr_filters(attr_filter_json)
    records, stats = scan_filtered_crop_records(
        image_dir, labels_dir, required_attrs=required
    )
    slim = [{"filename": r["filename"], "state": r["state"]} for r in records]
    print(f"Loaded {len(slim)} benign crops (stats={stats})")
    if full:
        return slim
    return stratified_sample(slim, max_per_state, seed=seed)

# loads attack records from a laser directory
def load_benign_records_from_laser(laser_dir: Path) -> list[dict[str, Any]]:
    csv_path = laser_dir / "results_ontology.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path}")
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variant = (row.get("variant") or "").strip().lower()
            path = row.get("path") or row.get("image") or ""
            if variant != "benign" and "benign_crops" not in path:
                continue
            if not path:
                continue
            rows.append(
                {
                    "path": path,
                    "filename": Path(path).name,
                    "state": row.get("gt_state") or "",
                    "variant": "benign",
                }
            )
    if not rows:
        raise RuntimeError(f"No benign rows in {csv_path}")
    return rows

# loads attack records from a csv file
def _load_attack_from_csv(laser_dir: Path) -> list[dict[str, Any]] | None:
    csv_path = laser_dir / "results_ontology.csv"
    if not csv_path.is_file():
        return None
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = row.get("path") or row.get("image") or ""
            if not path:
                continue
            rows.append(
                {
                    "path": path,
                    "state": row.get("gt_state") or "",
                    "variant": row.get("variant") or "",
                    "combo_key": row.get("combo_key") or "",
                }
            )
    return rows

# finds synth cache root under laser_dir or pointed by meta / manifest
def _resolve_synth_cache_dir(laser_dir: Path, meta: dict[str, Any] | None) -> Path | None:
    """Find synth cache root under laser_dir or pointed by meta / manifest."""
    for candidate in (
        laser_dir,
        laser_dir / "synth_jpg",
        laser_dir / "laser_cache",
    ):
        if (candidate / "manifest_entries.jsonl").is_file():
            return candidate
        if (candidate / "manifest.json").is_file():
            return candidate
    manifest = laser_dir / "manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if isinstance(data, dict) and data.get("synth_cache_dir"):
            cache = Path(data["synth_cache_dir"])
            if cache.is_dir():
                return cache
    if meta and meta.get("synth_cache_dir"):
        cache = Path(meta["synth_cache_dir"])
        if cache.is_dir():
            return cache
    return None

# loads attack records from a synth cache manifest
def _load_attack_from_manifest(cache_dir: Path) -> list[dict[str, Any]]:
    jsonl = cache_dir / "manifest_entries.jsonl"
    rows: list[dict[str, Any]] = []
    if jsonl.is_file():
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                path = entry.get("path") or entry.get("image") or ""
                if path and not os.path.isabs(path):
                    path = str(cache_dir / path)
                variant = (entry.get("variant") or "").strip().lower()
                if variant == "benign" or "benign_crops" in path:
                    continue
                if not path:
                    continue
                rows.append(
                    {
                        "path": path,
                        "state": entry.get("gt_state") or entry.get("benign_color") or "",
                        "variant": variant or "attack",
                        "combo_key": entry.get("combo_key") or "",
                    }
                )
        return rows
    # Fallback: thin / full manifest.json
    manifest = cache_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"No manifest under {cache_dir}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("entries") or []
    for entry in entries:
        path = entry.get("path") or entry.get("image") or ""
        if path and not os.path.isabs(path):
            path = str(cache_dir / path)
        variant = (entry.get("variant") or "").strip().lower()
        if variant == "benign" or "benign_crops" in str(path):
            continue
        if not path:
            continue
        rows.append(
            {
                "path": path,
                "state": entry.get("gt_state") or entry.get("benign_color") or "",
                "variant": variant or "attack",
                "combo_key": entry.get("combo_key") or "",
            }
        )
    return rows

# loads attack records from a laser directory or synth cache manifest
def load_attack_records(
    laser_dir: Path,
    *,
    meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Prefer results_ontology.csv; else synth-cache manifest under laser_dir / meta."""
    rows = _load_attack_from_csv(laser_dir)
    if rows is not None:
        print(f"Loaded {len(rows)} attack rows from {laser_dir / 'results_ontology.csv'}")
        return rows
    cache = _resolve_synth_cache_dir(laser_dir, meta)
    if cache is None:
        raise FileNotFoundError(
            f"No results_ontology.csv or synth cache manifest under {laser_dir}"
        )
    rows = _load_attack_from_manifest(cache)
    print(f"Loaded {len(rows)} attack rows from synth cache {cache}")
    return rows


# accumulates stats for a single record
def _accumulate_row_stats(stats: dict[str, Any], row: dict[str, Any]) -> None:
    pred_state = str(row.get("pred_state") or "")
    gt_state = str(row.get("gt_state") or "")
    card_ok = bool(row.get("card_ok"))
    invalid = bool(row.get("invalid", not card_ok))
    stats["n"] += 1
    stats["card_ok"] += int(card_ok)
    stats["invalid"] += int(invalid)
    stats["on_ok"] += int(bool(row.get("on_ok")))
    stats["color_ok"] += int(bool(row.get("color_ok")))
    stats["pred_consistent"] += int(bool(row.get("pred_consistent")))
    if gt_state and pred_state:
        if gt_state != pred_state:
            stats["n_wrong"] += 1
            stats["invalid_wrong"] += int(invalid)
        else:
            stats["n_correct"] += 1
            stats["invalid_correct"] += int(invalid)

# finalizes summary statistics
def finalize_summary(stats: dict[str, Any]) -> dict[str, Any]:
    n = max(stats["n"], 1)
    n_wrong = max(int(stats.get("n_wrong", 0)), 0)
    n_correct = max(int(stats.get("n_correct", 0)), 0)
    return {
        "n": stats["n"],
        "checker": "evidence_first_cie_xy_intensity",
        "pct_card_ok": 100.0 * stats["card_ok"] / n,
        "pct_invalid": 100.0 * stats.get("invalid", 0) / n,
        "pct_on_ok": 100.0 * stats.get("on_ok", 0) / n,
        "pct_color_ok": 100.0 * stats.get("color_ok", 0) / n,
        "pct_pred_consistent": 100.0 * stats.get("pred_consistent", 0) / n,
        "n_wrong": n_wrong,
        "n_correct": n_correct,
        "catch_rate": (
            100.0 * stats.get("invalid_wrong", 0) / n_wrong if n_wrong else None
        ),
        "false_alarm_rate": (
            100.0 * stats.get("invalid_correct", 0) / n_correct if n_correct else None
        ),
        "on_infer_mode": stats.get("on_infer_mode"),
        "module_color_counts": dict(stats.get("module_color_counts", {})),
    }


# evaluates cardinality consistency for a list of records
@torch.no_grad()
def eval_records(
    model: CardinalityHeads,
    records: list[dict[str, Any]],
    *,
    image_dir: str | None,
    meta: dict[str, Any],
    device: torch.device,
    batch_size: int,
    source: str,
    details_path: Path | None = None,
) -> dict[str, Any]:
    boxes = meta.get("adaptive_xy_boxes")
    module_crop_mode = str(meta.get("module_crop_mode") or "gray")
    intensity_on_thr = float(
        meta.get("module_on_intensity_threshold") or MODULE_ON_INTENSITY_THRESHOLD
    )
    use_intensity = bool(
        getattr(model, "has_module_intensity_head", False)
        and (
            meta.get("module_intensity_state_loss")
            or meta.get("mode") == "card_owned"
            or meta.get("train_mode") == "card_owned"
        )
    )
    # Fall back: if intensity head exists, use it (card_owned checkpoints).
    if not use_intensity and getattr(model, "has_module_intensity_head", False):
        use_intensity = True
    cd_map = CdMap.from_dict(meta.get("module_cd_map"))
    if cd_map is not None:
        print(
            f"Loaded module cd_map: a={cd_map.a:.4f}, b={cd_map.b:.4f}, "
            f"gate={cd_map.gate_threshold:.4f}"
        )

    details_fh = None
    if details_path is not None:
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_fh = open(details_path, "w", encoding="utf-8")

    stats: dict[str, Any] = {
        "n": 0,
        "card_ok": 0,
        "invalid": 0,
        "on_ok": 0,
        "color_ok": 0,
        "pred_consistent": 0,
        "n_wrong": 0,
        "n_correct": 0,
        "invalid_wrong": 0,
        "invalid_correct": 0,
        "module_color_counts": defaultdict(int),
        "on_infer_mode": "intensity_threshold" if use_intensity else "all_on_fallback",
    }

    def flush_batch(
        batch_recs: list[dict[str, Any]],
        tensors: list[torch.Tensor],
        color_mod_tensors: list[torch.Tensor],
        intensity_mod_tensors: list[torch.Tensor],
    ):
        if not tensors:
            return
        x = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        logits, _bright, _feats = model(x)
        pred_idx = logits.float().argmax(dim=-1).cpu().numpy()

        color_feats = None
        if getattr(model, "has_xy_head", False):
            mod_x = torch.stack(color_mod_tensors, dim=0).to(device, non_blocking=True)
            color_feats = model.encode_modules(mod_x)

        intensity_feats = None
        intensity_np = None
        on_mask_np = None
        if use_intensity:
            if module_crop_mode == "color" and color_feats is not None:
                intensity_feats = color_feats
            else:
                int_x = torch.stack(intensity_mod_tensors, dim=0).to(
                    device, non_blocking=True
                )
                intensity_feats = model.encode_modules(int_x)
            intensity_pred = module_intensity_from_feats(
                intensity_feats.float(),
                model.module_intensity_head,
                cd_map=cd_map,
            )
            intensity_np = intensity_pred.detach().cpu().numpy()
            on_mask_np = infer_on_mask_from_intensity(
                intensity_pred, threshold=intensity_on_thr
            ).detach().cpu().numpy()

        xy_pred_np = None
        if color_feats is not None and getattr(model, "has_xy_head", False):
            xy_pred_np = (
                color_xy_from_module_feats(color_feats.float(), model.xy_head)
                .cpu()
                .numpy()
                .reshape(-1, 3, 2)
            )

        for j, rec in enumerate(batch_recs):
            pred_state = (
                CANON_STATES[int(pred_idx[j])]
                if int(pred_idx[j]) < len(CANON_STATES)
                else "off"
            )
            if on_mask_np is not None:
                on_mask = [bool(on_mask_np[j, mi]) for mi in range(3)]
            else:
                # Baseline without intensity head: treat all modules as candidates
                # via color-only (degenerate); prefer all-off if no xy either.
                on_mask = [True, True, True] if xy_pred_np is not None else [False, False, False]

            module_colors: dict[str, str] = {}
            for mi, mname in enumerate(MODULE_ORDER):
                if xy_pred_np is None or not np.isfinite(xy_pred_np[j, mi]).all():
                    module_colors[mname] = "other"
                else:
                    lab, _ = label_module_color_xy(
                        float(xy_pred_np[j, mi, 0]),
                        float(xy_pred_np[j, mi, 1]),
                        boxes,
                        prefer_box=True,
                    )
                    module_colors[mname] = lab

            if on_mask_np is None:
                on_mask = [
                    module_colors.get(MODULE_ORDER[i], "other") in MODULE_COLORS
                    for i in range(3)
                ]

            for c in module_colors.values():
                stats["module_color_counts"][c] += 1

            module_intensities = None
            if intensity_np is not None:
                module_intensities = {
                    MODULE_ORDER[mi]: float(intensity_np[j, mi]) for mi in range(3)
                }
            module_xy = None
            if xy_pred_np is not None:
                module_xy = {
                    MODULE_ORDER[mi]: [
                        float(xy_pred_np[j, mi, 0]),
                        float(xy_pred_np[j, mi, 1]),
                    ]
                    for mi in range(3)
                }

            ev = apply_evidence_metrics(
                on_mask=on_mask,
                module_colors=module_colors,
                pred_state=pred_state,
                module_intensities=module_intensities,
                module_xy=module_xy,
                intensity_threshold=intensity_on_thr if intensity_np is not None else None,
            )
            gt_state = str(rec.get("state") or "")
            row = {
                "source": source,
                "gt_state": gt_state,
                "pred_state": pred_state,
                "evidence_state": ev["evidence_state"],
                "on_mask": {
                    "top": on_mask[0],
                    "middle": on_mask[1],
                    "bottom": on_mask[2],
                },
                "module_colors": module_colors,
                "card_ok": ev["card_ok"],
                "on_ok": ev["on_ok"],
                "color_ok": ev["color_ok"],
                "pred_consistent": ev["pred_consistent"],
                "invalid": ev["invalid"],
                "explanation": ev["explanation"],
            }
            if module_intensities is not None:
                row["module_intensities"] = module_intensities
                row["module_on_intensity_threshold"] = intensity_on_thr
            if module_xy is not None:
                row["module_xy"] = module_xy
            if "filename" in rec:
                row["filename"] = rec["filename"]
            if "path" in rec:
                row["path"] = rec["path"]
            if details_fh is not None:
                details_fh.write(json.dumps(row) + "\n")
            _accumulate_row_stats(stats, row)

    batch_recs: list[dict[str, Any]] = []
    batch_tensors: list[torch.Tensor] = []
    batch_color_mod: list[torch.Tensor] = []
    batch_int_mod: list[torch.Tensor] = []

    for rec in tqdm(records, desc=f"card-{source}", leave=False):
        try:
            if rec.get("path"):
                path = Path(rec["path"])
            elif image_dir and rec.get("filename"):
                path = Path(image_dir) / rec["filename"]
            else:
                continue
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        batch_recs.append(rec)
        batch_tensors.append(model.preprocess(img))
        module_pils = split_module_pils(img)
        batch_color_mod.append(
            torch.stack([model.preprocess(mp) for mp in module_pils])
        )
        if module_crop_mode == "gray":
            int_pils = [pil_to_gray_rgb(mp) for mp in module_pils]
        else:
            int_pils = module_pils
        batch_int_mod.append(
            torch.stack([model.preprocess(mp) for mp in int_pils])
        )
        if len(batch_tensors) >= batch_size:
            flush_batch(batch_recs, batch_tensors, batch_color_mod, batch_int_mod)
            batch_recs, batch_tensors = [], []
            batch_color_mod, batch_int_mod = [], []
    flush_batch(batch_recs, batch_tensors, batch_color_mod, batch_int_mod)

    if details_fh is not None:
        details_fh.close()
    return finalize_summary(stats)


# CLI args
def main() -> None:
    p = argparse.ArgumentParser(
        description="Lean cardinality consistency eval (baseline + card_owned)"
    )
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--meta", type=str, default=None)
    p.add_argument(
        "--source", type=str, default="benign", choices=["benign", "attack", "both"]
    )
    p.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    p.add_argument("--labels_dir", type=str, default=DEFAULT_LABELS_DIR)
    p.add_argument("--laser_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_per_state", type=int, default=2000)
    p.add_argument("--full", action="store_true")
    p.add_argument("--attr_filter_json", type=str, default=None)
    p.add_argument(
        "--benign_from_laser",
        action="store_true",
        help="Load benign frames from laser_dir/results_ontology.csv",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--write_details", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    meta: dict[str, Any] = {}
    if args.meta and Path(args.meta).is_file():
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))

    model = load_model(Path(args.ckpt), device)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    sources = ["benign", "attack"] if args.source == "both" else [args.source]
    all_summaries = []

    for src in sources:
        if src == "benign":
            if args.benign_from_laser:
                if not args.laser_dir:
                    raise SystemExit("--benign_from_laser requires --laser_dir")
                records = load_benign_records_from_laser(Path(args.laser_dir))
            else:
                records = load_benign_records(
                    args.image_dir,
                    args.labels_dir,
                    max_per_state=None if args.full else args.max_per_state,
                    full=args.full,
                    seed=args.seed,
                    attr_filter_json=args.attr_filter_json,
                )
            image_dir = args.image_dir
        else:
            if not args.laser_dir:
                raise SystemExit("--source attack/both requires --laser_dir")
            records = load_attack_records(Path(args.laser_dir), meta=meta)
            image_dir = None

        details_path = out_root / src / "details.jsonl" if args.write_details else None
        if details_path:
            details_path.parent.mkdir(parents=True, exist_ok=True)
        summary = eval_records(
            model,
            records,
            image_dir=image_dir,
            meta=meta,
            device=device,
            batch_size=args.batch_size,
            source=src,
            details_path=details_path,
        )
        src_out = out_root / src
        src_out.mkdir(parents=True, exist_ok=True)
        with open(src_out / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"source": src, **summary}, f, indent=2)
        catch = summary.get("catch_rate")
        far = summary.get("false_alarm_rate")
        print(
            f"[{src}] n={summary['n']}  card_ok={summary['pct_card_ok']:.1f}%  "
            f"invalid={summary['pct_invalid']:.1f}%  "
            f"catch={catch if catch is not None else 'n/a'}  "
            f"FAR={far if far is not None else 'n/a'}"
        )
        all_summaries.append({"source": src, **summary})

    with open(out_root / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Wrote {out_root / 'leaderboard.json'}")


if __name__ == "__main__":
    main()
