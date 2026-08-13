"""Shared ontology / module helpers for train + cardinality eval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from helper import CANON_STATES

# module / ontology constants

MODULE_ORDER = ("top", "middle", "bottom")
TOP, MIDDLE, BOTTOM = 0, 1, 2
MODULE_IDX = {"top": TOP, "middle": MIDDLE, "bottom": BOTTOM}
NUM_MODULES = 3

TRAFFIC_COLORS = ("red", "yellow", "green")
MODULE_COLORS = TRAFFIC_COLORS
MODULE_SLOT_COLOR = {"top": "red", "middle": "yellow", "bottom": "green"}

STATE_ON_MASK = {
    "red": (1.0, 0.0, 0.0),
    "yellow": (0.0, 1.0, 0.0),
    "green": (0.0, 0.0, 1.0),
    "red_yellow": (1.0, 1.0, 0.0),
    "off": (0.0, 0.0, 0.0),
}

# color-only module specs (state→slot and xy / cardinality masks)
ONTOLOGY_MODULE_SPECS = {
    "red": {"top": {"color": "red"}},
    "yellow": {"middle": {"color": "yellow"}},
    "green": {"bottom": {"color": "green"}},
    "red_yellow": {
        "top": {"color": "red"},
        "middle": {"color": "yellow"},
    },
}

# ideal CIE xy coordinates - calculated from rdf
IDEAL_XY: dict[str, tuple[float, float]] = {
    "red": (0.7079177916216642, 0.2920271089348397),
    "yellow": (0.5751513113651647, 0.4242322349249047),
    "green": (0.0038585209003215433, 0.6548231511254019),
}

# luminance intensity threshold - defined by rdf
MODULE_ON_INTENSITY_THRESHOLD = 200.0
DEFAULT_MODULE_INTENSITY_RANGES: dict[str, tuple[float, float]] = {
    "off": (0.0, 0.0),
    "red": (200.0, 300.0),
    "yellow": (250.0, 400.0),
    "green": (300.0, 400.0),
}

MODULE_STATE_NAMES = ("off", "red", "yellow", "green")
MODULE_STATE_TO_IDX = {n: i for i, n in enumerate(MODULE_STATE_NAMES)}
NUM_MODULE_STATES = len(MODULE_STATE_NAMES)

CD_MAP_OFF_TARGET = 100.0
CD_MAP_ON_TARGET = 300.0
CARDINALITY_STATES = ("red", "yellow", "green", "red_yellow", "off")

STATE_TO_IDX = {state: i for i, state in enumerate(CANON_STATES)}
NUM_CLASSES = len(CANON_STATES)


# module crop helpers

# splits a crop into equal vertical thirds
def split_three_zones(roi: np.ndarray) -> dict[str, np.ndarray]:
    h = int(roi.shape[0])
    y1 = h // 3
    y2 = (2 * h) // 3
    return {
        "top": roi[:y1, :],
        "middle": roi[y1:y2, :],
        "bottom": roi[y2:, :],
    }


# splits a traffic-light crop into top/mid/bottom PILs
def split_module_pils(image):
    arr = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
    zones = split_three_zones(arr)
    return [Image.fromarray(np.ascontiguousarray(zones[name])) for name in MODULE_ORDER]


# converts a PIL image to 3-channel grayscale
def pil_to_gray_rgb(image):
    if not hasattr(image, "convert"):
        raise TypeError("pil_to_gray_rgb expects a PIL image")
    return image.convert("L").convert("RGB")


# CdMap class - maps relative brightness to luminous intensity
@dataclass
class CdMap:
    """Affine map I = a + b * r (frozen after fit)."""

    a: float
    b: float
    r_off_mean: float
    r_on_mean: float
    gate_threshold: float

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        return float(self.a) + float(self.b) * r

    def to_dict(self) -> dict[str, float]:
        return {
            "a": self.a,
            "b": self.b,
            "r_off_mean": self.r_off_mean,
            "r_on_mean": self.r_on_mean,
            "gate_threshold": self.gate_threshold,
            "off_target_cd": CD_MAP_OFF_TARGET,
            "on_target_cd": CD_MAP_ON_TARGET,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CdMap | None":
        if not d:
            return None
        return cls(
            a=float(d["a"]),
            b=float(d["b"]),
            r_off_mean=float(d.get("r_off_mean", 0.0)),
            r_on_mean=float(d.get("r_on_mean", 0.0)),
            gate_threshold=float(d.get("gate_threshold", 1.0)),
        )


# per-module colors for a light state
def module_states_from_light_state(light_state: str) -> tuple[str, str, str]:
    modules = ONTOLOGY_MODULE_SPECS.get(light_state, {})
    out = ["off", "off", "off"]
    for module_name, spec in modules.items():
        idx = MODULE_ORDER.index(module_name)
        out[idx] = str(spec["color"])
    return out[0], out[1], out[2]


# predicts CIE xy per module from shared features
def color_xy_from_module_feats(
    module_feats: torch.Tensor,
    xy_head: nn.Module,
) -> torch.Tensor:
    if module_feats.dim() == 3:
        B, M, D = module_feats.shape
        flat = module_feats.reshape(B * M, D)
    else:
        flat = module_feats
        B = flat.shape[0] // NUM_MODULES
    return xy_head(flat).reshape(B, NUM_MODULES * 2)


# module intensity predictions from shared head features
def module_intensity_from_feats(
    module_feats: torch.Tensor,
    module_intensity_head: nn.Module,
    *,
    cd_map: CdMap | None = None,
    return_logits: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if module_feats.dim() == 3:
        B, M, D = module_feats.shape
        flat = module_feats.reshape(B * M, D)
    else:
        flat = module_feats
        B = flat.shape[0] // NUM_MODULES
        M = NUM_MODULES
    raw = module_intensity_head(flat).reshape(B, M)
    pred_r = F.softplus(raw)
    intensity = cd_map(pred_r).clamp(min=0.0) if cd_map is not None else pred_r
    if return_logits:
        return intensity, raw
    return intensity


# infers on mask from module intensity
def infer_on_mask_from_intensity(
    intensity: torch.Tensor,
    *,
    threshold: float = MODULE_ON_INTENSITY_THRESHOLD,
) -> torch.Tensor:
    return intensity >= float(threshold)
