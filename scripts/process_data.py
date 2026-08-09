#!/usr/bin/env python3
"""
Build DTLD training crops + inference attack cache from a dataset root.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from helper import (
    CANON_STATES,
    DEFAULT_ATTR_FILTERS,
    DEFAULT_DTLD_ROOT,
    DEFAULT_LABELS_DIR,
    DEFAULT_LASER_TRACES,
    DEFAULT_PROCESSED_ROOT,
    JPEG_QUALITY,
    attributes_match,
    convert_tiff_to_frame,
    crop_frame,
    discover_city_jsons,
    ensure_dir,
    load_attr_filters,
    normalize_rel_path,
    normalize_state,
    resolve_base,
    resolve_tiff_path,
)

# loads the simulation module
def _load_sim_module():
    """Import sibling ``test_models.py`` for synth-cache helpers."""
    path = Path(__file__).resolve().parent / "test_models.py"
    spec = importlib.util.spec_from_file_location("test_models", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

# builds color crops from the DTLD dataset

def build_crops_jpg(
    *,
    dtld_root: str,
    labels_dir: str,
    crops_dir: Path,
    required_attrs: dict[str, str],
    exclude_unknown: bool = True,
    pad: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    """
    Demosaic DTLD TIFFs and write attr-filtered color crops.

    Same crop naming as ``modified_pipeline`` so training JSON joins still work.
    """
    dtld_root = resolve_base(dtld_root)
    json_paths = discover_city_jsons(dtld_root, labels_dir=labels_dir)
    if not json_paths:
        raise FileNotFoundError(
            f"No per-city DTLD JSON files under labels_dir={labels_dir!r}"
        )

    if crops_dir.is_dir() and any(crops_dir.rglob("*.jpg")) and not force:
        n_existing = sum(1 for _ in crops_dir.rglob("*.jpg"))
        print(
            f"[skip crops] {crops_dir} already has {n_existing} JPGs "
            "(pass --force to rebuild)"
        )
        return {
            "skipped": True,
            "crops_dir": str(crops_dir),
            "n_crops": n_existing,
            "json_paths": json_paths,
        }

    if force and crops_dir.is_dir():
        print(f"[force] clearing {crops_dir}")
        shutil.rmtree(crops_dir)
    ensure_dir(crops_dir)

    print(f"[crops] mode=color demosaic  pad={pad}")
    print(f"[crops] attribute filter: {required_attrs}")
    print(f"[crops] exclude_unknown={exclude_unknown}")
    print(f"[crops] {len(json_paths)} city JSON(s) → {crops_dir}")

    stats: dict[str, int] = defaultdict(int)
    stats["n_json"] = len(json_paths)

    for jp in json_paths:
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
        images = data.get("images", [])
        json_tag = Path(jp).stem
        print(f"[crops] {Path(jp).name}: {len(images)} frames")

        for idx_img, entry in enumerate(images):
            rel = entry.get("image_path") or entry.get("disparity_image_path")
            if not rel:
                stats["skip_no_path"] += 1
                continue
            rel_norm = normalize_rel_path(rel)
            tiff_path = resolve_tiff_path(dtld_root, rel_norm)
            if not os.path.isfile(tiff_path):
                stats["missing_tiff"] += 1
                continue

            frame = convert_tiff_to_frame(tiff_path, use_color=True)
            if frame is None:
                stats["bad_frame"] += 1
                continue

            H_frame, W_frame = frame.shape[:2]
            labels = entry.get("labels", [])
            wrote_any = False

            for box_idx, lb in enumerate(labels):
                x, y, w, h = lb.get("x"), lb.get("y"), lb.get("w"), lb.get("h")
                if None in (x, y, w, h):
                    stats["skip_bad_box"] += 1
                    continue

                attrs = lb.get("attributes", {}) or {}
                state = normalize_state(attrs.get("state"))
                if exclude_unknown and (state is None or state not in CANON_STATES):
                    stats["skip_unknown"] += 1
                    continue
                if required_attrs and not attributes_match(attrs, required_attrs):
                    stats["skip_attr"] += 1
                    continue

                crop_pil, (x0, y0, x1, y1) = crop_frame(
                    frame, x, y, w, h, W_frame, H_frame, pad=pad
                )
                state_tag = state if state else "unknown"
                dst_dir = crops_dir / state_tag
                ensure_dir(dst_dir)
                fname = (
                    f"{json_tag}__img{idx_img:05d}__box{box_idx:03d}__"
                    f"{state_tag}__{x0}_{y0}_{x1}_{y1}.jpg"
                )
                out_path = dst_dir / fname
                try:
                    crop_pil.save(out_path, format="JPEG", quality=JPEG_QUALITY)
                except Exception as exc:
                    print(f"[WARN] Failed to save {out_path}: {exc}")
                    stats["save_failed"] += 1
                    continue
                stats["n_crops"] += 1
                stats[f"state_{state_tag}"] += 1
                wrote_any = True

            if wrote_any:
                stats["frames_used"] += 1
            del frame

    manifest = {
        "dtld_root": dtld_root,
        "labels_dir": labels_dir,
        "crops_dir": str(crops_dir),
        "required_attributes": required_attrs,
        "exclude_unknown": exclude_unknown,
        "pad": pad,
        "color_crops": True,
        "stats": dict(stats),
        "state_counts": {
            s: int(stats.get(f"state_{s}", 0)) for s in CANON_STATES
        },
    }
    man_path = crops_dir / "crops_meta.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[crops] wrote {stats['n_crops']} crops → {crops_dir}")
    print(f"[crops] state counts: {manifest['state_counts']}")
    print(f"[crops] meta → {man_path}")
    return manifest

# builds a synth cache from the DTLD dataset
def build_synth_jpg(
    *,
    dtld_root: str,
    synth_dir: Path,
    red_laser_dir: str,
    green_laser_dir: str,
    ir_laser_dir: str,
    required_attrs: dict[str, str],
    exclude_unknown: bool = True,
    dtld_split: str = "test",
    dtld_labels: str | None = None,
    pad: int = 4,
    max_per_class: int | None = None,
    all_attack_combos: bool = False,
    realistic_only: bool = False,
    force: bool = False,
    max_save_benign: int = 10,
    max_save_attack: int = 10,
) -> dict[str, Any]:
    """
    Build a ``test_models``-compatible synth cache under synth_dir.

    Benign crops come from attr-filtered DTLD TIFF demosaic (official split);
    attacks are pasted into off modules only.
    """
    sim = _load_sim_module()
    synth_dir = Path(synth_dir).resolve()
    ensure_dir(synth_dir)

    eval_default = not all_attack_combos and not realistic_only
    fingerprint = {
        "benign_source": "dtld",
        "realistic_only": bool(realistic_only),
        "benign_only": False,
        "all_attack_combos": bool(all_attack_combos),
        "eval_default": bool(eval_default),
        "dtld_root": os.path.abspath(dtld_root),
        "dtld_split": dtld_split,
        "dtld_labels": os.path.abspath(dtld_labels) if dtld_labels else None,
        "dtld_max_per_class": max_per_class,
        "max_per_class": max_per_class,
        "dtld_pad": pad,
        "dtld_attr_filters": required_attrs,
        "dtld_exclude_unknown": exclude_unknown,
        "red_laser_dir": os.path.abspath(red_laser_dir),
        "green_laser_dir": os.path.abspath(green_laser_dir),
        "ir_laser_dir": (
            None if realistic_only else os.path.abspath(ir_laser_dir)
        ),
        "writer": "scripts/process_data.py",
    }

    if sim.synth_cache_ready(str(synth_dir)) and not force:
        print(
            f"[skip synth] cache ready at {synth_dir} "
            "(pass --force to rebuild)"
        )
        return {"skipped": True, "synth_dir": str(synth_dir), **fingerprint}

    if force and synth_dir.is_dir():
        print(f"[force] clearing {synth_dir}")
        shutil.rmtree(synth_dir)
        ensure_dir(synth_dir)

    mode = (
        "realistic (2 combos)"
        if realistic_only
        else (
            "full off-module matrix"
            if all_attack_combos
            else "eval-default curated combos"
        )
    )
    print(f"[synth] DTLD split={dtld_split}  attacks={mode}")
    print(f"[synth] attribute filter: {required_attrs}")
    print(f"[synth] → {synth_dir}")

    for label, path in [
        ("red_laser_dir", red_laser_dir),
        ("green_laser_dir", green_laser_dir),
    ] + ([] if realistic_only else [("ir_laser_dir", ir_laser_dir)]):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # Dummy benign dirs unused when benign_source='dtld'.
    dummy = str(synth_dir)
    sim.generate_simulated_dataset(
        green_benign_dir=dummy,
        red_benign_dir=dummy,
        red_laser_dir=red_laser_dir,
        green_laser_dir=green_laser_dir,
        ir_laser_dir=ir_laser_dir,
        output_dir=str(synth_dir),
        max_save_benign=max_save_benign,
        max_save_attack=max_save_attack,
        benign_source="dtld",
        dtld_root=dtld_root,
        dtld_split=dtld_split,
        dtld_labels=dtld_labels,
        dtld_max_per_class=max_per_class,
        max_per_class=max_per_class,
        dtld_pad=pad,
        dtld_attr_filters=required_attrs,
        dtld_exclude_unknown=exclude_unknown,
        realistic_only=realistic_only,
        eval_default=eval_default,
        benign_only=False,
        return_entries=False,
    )
    sim.finalize_synth_cache(str(synth_dir), fingerprint, return_entries=False)
    print(f"[synth] done → {synth_dir}")
    return {"skipped": False, "synth_dir": str(synth_dir), **fingerprint}

# cli
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create crops_jpg (color training crops) and synth_jpg "
            "(premade laser-attack cache) from DTLD."
        )
    )
    p.add_argument("--dtld_root", default=DEFAULT_DTLD_ROOT)
    p.add_argument("--labels_dir", default=DEFAULT_LABELS_DIR)
    p.add_argument(
        "--out_root",
        default=DEFAULT_PROCESSED_ROOT,
        help="Creates <out_root>/crops_jpg and <out_root>/synth_jpg",
    )
    p.add_argument(
        "--crops_dir",
        default=None,
        help="Override crops output (default: <out_root>/crops_jpg)",
    )
    p.add_argument(
        "--synth_dir",
        default=None,
        help="Override synth cache output (default: <out_root>/synth_jpg)",
    )

    p.add_argument("--crops_only", action="store_true")
    p.add_argument("--synth_only", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if outputs already exist",
    )

    p.add_argument(
        "--attr_filter_json",
        default=None,
        help=(
            "Optional JSON of required label attributes. "
            f"Default: {DEFAULT_ATTR_FILTERS}"
        ),
    )
    p.add_argument(
        "--no_attr_filter",
        action="store_true",
        help="Keep all labeled boxes (still excludes unknown unless --include_unknown)",
    )
    p.add_argument(
        "--include_unknown",
        action="store_true",
        help="Keep unknown-state labels in crops (synth still uses known states)",
    )
    p.add_argument("--pad", type=int, default=4)

    # Synth options
    p.add_argument(
        "--dtld_split",
        default="test",
        choices=["train", "test", "all"],
        help="Official DTLD split used for synth_jpg benign crops (default: test)",
    )
    p.add_argument(
        "--dtld_labels",
        default=None,
        help="Optional override for DTLD_train/test/all.json path",
    )
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument(
        "--all_attack_combos",
        action="store_true",
        help="Full off-module × red/green/ir matrix (default: curated eval combos)",
    )
    p.add_argument("--realistic_only", action="store_true")
    p.add_argument(
        "--laser_root",
        default=DEFAULT_LASER_TRACES,
        help="Parent of Red / Green / IR_modulated laser-trace dirs",
    )
    p.add_argument("--red_laser_dir", default=None)
    p.add_argument("--green_laser_dir", default=None)
    p.add_argument("--ir_laser_dir", default=None)
    p.add_argument("--max_save_benign", type=int, default=10)
    p.add_argument("--max_save_attack", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.crops_only and args.synth_only:
        raise SystemExit("Use at most one of --crops_only / --synth_only")

    out_root = Path(args.out_root).resolve()
    crops_dir = Path(args.crops_dir).resolve() if args.crops_dir else out_root / "crops_jpg"
    synth_dir = Path(args.synth_dir).resolve() if args.synth_dir else out_root / "synth_jpg"
    ensure_dir(out_root)

    if args.no_attr_filter:
        required_attrs: dict[str, str] = {}
    else:
        required_attrs = load_attr_filters(args.attr_filter_json)

    laser_root = args.laser_root
    red_laser = args.red_laser_dir or os.path.join(laser_root, "Red")
    green_laser = args.green_laser_dir or os.path.join(laser_root, "Green")
    ir_laser = args.ir_laser_dir or os.path.join(laser_root, "IR_modulated")

    do_crops = not args.synth_only
    do_synth = not args.crops_only

    print("=" * 60)
    print(f"out_root = {out_root}")
    print(f"crops    = {crops_dir}  ({'on' if do_crops else 'skip'})")
    print(f"synth    = {synth_dir}  ({'on' if do_synth else 'skip'})")
    print(f"attrs    = {required_attrs or '(none)'}")
    print("=" * 60)

    crops_meta = None
    if do_crops:
        if not os.path.isdir(args.dtld_root):
            raise FileNotFoundError(f"--dtld_root not found: {args.dtld_root}")
        if not os.path.isdir(args.labels_dir):
            raise FileNotFoundError(f"--labels_dir not found: {args.labels_dir}")
        print("\n=== Step A: crops_jpg ===")
        crops_meta = build_crops_jpg(
            dtld_root=args.dtld_root,
            labels_dir=args.labels_dir,
            crops_dir=crops_dir,
            required_attrs=required_attrs,
            exclude_unknown=not args.include_unknown,
            pad=args.pad,
            force=args.force,
        )

    synth_meta = None
    if do_synth:
        if not os.path.isdir(args.dtld_root):
            raise FileNotFoundError(f"--dtld_root not found: {args.dtld_root}")
        print("\n=== Step B: synth_jpg ===")
        synth_meta = build_synth_jpg(
            dtld_root=args.dtld_root,
            synth_dir=synth_dir,
            red_laser_dir=red_laser,
            green_laser_dir=green_laser,
            ir_laser_dir=ir_laser,
            required_attrs=required_attrs,
            exclude_unknown=not args.include_unknown,
            dtld_split=args.dtld_split,
            dtld_labels=args.dtld_labels,
            pad=args.pad,
            max_per_class=args.max_per_class,
            all_attack_combos=args.all_attack_combos,
            realistic_only=args.realistic_only,
            force=args.force,
            max_save_benign=args.max_save_benign,
            max_save_attack=args.max_save_attack,
        )

    summary = {
        "out_root": str(out_root),
        "crops_dir": str(crops_dir),
        "synth_dir": str(synth_dir),
        "crops": crops_meta,
        "synth": synth_meta,
    }
    summary_path = out_root / "process_data_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[DONE] summary → {summary_path}")


if __name__ == "__main__":
    main()
