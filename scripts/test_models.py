"""
Generate simulated laser-injection attack images and evaluate VLMs.
"""

import argparse
import csv
import gc
import itertools
import json
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

# helper functions for the DTLD dataset
from helper import (
    DEFAULT_ATTR_FILTERS,
    DEFAULT_EVAL_ATTACK_COMBOS,
    DEFAULT_LASER_TRACES,
    DTLD_SPLITS,
    INJECTABLE_LASER_COLORS,
    eval_injections_by_state,
    load_attr_filters,
    load_dtld_benign_crop_items,
    off_modules_for_state,
    paste_laser_pattern,
    valid_laser_injections,
)
# model inference functions
from model_inference import (
    BatchImagePrefetcher,
    CANON_STATES,
    DEFAULT_MODULE_ANALYSIS_METHOD,
    MODULE_ANALYSIS_METHODS,
    compute_state_metrics,
    load_model_and_classifier,
    predict_batch,
    print_comparison,
    print_state_metrics_block,
)

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# reference laser image
LASER_REF_FILENAME = "2024-09-10_00-49-56-478947.jpg"
LASER_MASK_THRESHOLD = 150
SIM_BRIGHT_THRESHOLD = 100
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# benign light states used for attack synthesis
BENIGN_COLORS = ("green", "red")
DTLD_BENIGN_COLORS = ("green", "red", "yellow", "off", "red_yellow")
LASER_COLORS = tuple(INJECTABLE_LASER_COLORS)  # red, green, ir
REALISTIC_INJECTIONS = (
    ("green", "red", "top"),      # green lit -> inject red into off top
    ("red", "green", "bottom"),   # red lit -> inject green into off bottom
)
BENIGN_GT_STATE = {
    "green": "green",
    "red": "red",
    "yellow": "yellow",
    "off": "off",
    "red_yellow": "red_yellow",
}
BENIGN_SOURCES = ("sim", "dtld")

# Preferred laser references (brightest frames) when present.
LASER_REF_CANDIDATES = (
    LASER_REF_FILENAME,
    "mag_7.png",
    "mag_7.jpg",
)

# lists image files in a directory
def list_image_files(directory):
    return sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    )

# extracts bright colors from an image
def extract_bright_colors(image_path, threshold=75):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, threshold])
    upper_bound = np.array([179, 255, 255])
    return cv2.inRange(hsv_image, lower_bound, upper_bound)

# gets bounding boxes from a mask
def get_bounding_boxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x + w, y + h))
    return bounding_boxes

# gets the path to the reference laser image
def _laser_reference_path(laser_dir):
    """Prefer a known bright reference; avoid sorted-first near-black frames."""
    for name in LASER_REF_CANDIDATES:
        candidate = os.path.join(laser_dir, name)
        if os.path.isfile(candidate):
            return candidate

    files = list_image_files(laser_dir)
    if not files:
        raise FileNotFoundError(f"No laser images found in {laser_dir}")

    # Fall back to the brightest file under the mask threshold.
    best_path, best_score = None, -1
    for fname in files:
        path = os.path.join(laser_dir, fname)
        image = cv2.imread(path)
        if image is None:
            continue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        score = int((hsv[:, :, 2] >= LASER_MASK_THRESHOLD).sum())
        if score > best_score:
            best_score = score
            best_path = path
    if best_path is None:
        return os.path.join(laser_dir, files[0])
    return best_path

# builds attack patterns from a laser directory
def build_attack_patterns(laser_dir, mask_threshold=LASER_MASK_THRESHOLD):
    """Extract bright laser patterns using a shared mask from the reference image."""
    img_paths = [os.path.join(laser_dir, f) for f in list_image_files(laser_dir)]
    if not img_paths:
        raise FileNotFoundError(f"No laser images found in {laser_dir}")

    ref_path = _laser_reference_path(laser_dir)
    mask = extract_bright_colors(ref_path, threshold=mask_threshold)
    # If the shared mask is empty, self-mask each frame.
    use_shared = int(mask.sum()) > 0
    patterns = []
    pattern_names = []
    for im_path in img_paths:
        image = cv2.imread(im_path)
        if image is None:
            raise ValueError(f"Could not read laser image: {im_path}")
        if use_shared:
            pattern = cv2.bitwise_and(image, image, mask=mask)
        else:
            self_mask = extract_bright_colors(im_path, threshold=mask_threshold)
            pattern = cv2.bitwise_and(image, image, mask=self_mask)
        # Skip near-black extracts (common for early Red/Green intensity frames).
        if int(pattern.max()) < 32:
            continue
        patterns.append(pattern)
        pattern_names.append(os.path.basename(im_path))
    if not patterns:
        raise ValueError(f"No usable (non-dark) laser patterns in {laser_dir}")
    return patterns, pattern_names

# placement label
def placement_display(placement):
    """placement label"""
    return {
        "opposite": "opposite",
        "on_top": "on top",
        "top": "top",
        "middle": "middle",
        "bottom": "bottom",
    }.get(placement, placement)

# combo key
def combo_key(benign_color, laser_color, target_module):
    return f"{laser_color}_laser_on_{benign_color}_{target_module}"

# attack injections for a mode
def attack_injections_for_mode(benign_color, realistic_only=False, eval_default=False):
    """
    Valid (laser_color, target_module) pairs for a benign state.
    eval_default: realistic DEFAULT_EVAL_ATTACK_COMBOS.
    """
    if realistic_only:
        return [
            (laser, module)
            for state, laser, module in REALISTIC_INJECTIONS
            if state == benign_color
        ]
    if eval_default:
        return list(eval_injections_by_state().get(benign_color, []))
    return valid_laser_injections(benign_color, laser_colors=LASER_COLORS)

# laser colors for a mode   
def laser_colors_for_mode(realistic_only=False, eval_default=False):
    if realistic_only:
        return ("red", "green")
    if eval_default:
        lasers = {
            laser
            for pairs in eval_injections_by_state().values()
            for laser, _ in pairs
        }
        return tuple(c for c in LASER_COLORS if c in lasers)
    return LASER_COLORS

# resolves the attack combo allowlist
def resolve_attack_combo_allowlist(args) -> set[str] | None:
    """Return combo_key allowlist for eval, or None = keep all attacks.

    Default: DEFAULT_EVAL_ATTACK_COMBOS (notebook example / realistic eval mask).
    ``--all_attack_combos`` disables the filter.
    ``--realistic_only`` uses the 2-combo legacy realistic set.
    ``--attack_combos`` accepts comma-separated keys or a JSON list file.
    """
    if bool(getattr(args, "all_attack_combos", False)):
        return None
    raw = getattr(args, "attack_combos", None)
    if raw:
        path = Path(str(raw))
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "combos" in data:
                data = data["combos"]
            if not isinstance(data, (list, tuple)):
                raise ValueError(f"--attack_combos file must be a JSON list: {path}")
            return {str(x) for x in data}
        return {x.strip() for x in str(raw).split(",") if x.strip()}
    if bool(getattr(args, "realistic_only", False)):
        return {
            combo_key(state, laser, module)
            for state, laser, module in REALISTIC_INJECTIONS
        }
    return set(DEFAULT_EVAL_ATTACK_COMBOS)

# checks if a manifest entry is allowed
def manifest_entry_allowed(entry: dict, allowlist: set[str] | None) -> bool:
    """Keep all benign rows; filter attacks by combo_key allowlist."""
    if allowlist is None:
        return True
    if entry.get("variant") != "attack":
        return True
    return str(entry.get("combo_key") or "") in allowlist

# caps the entries per attack combo
def cap_entries_per_attack_combo(
    entries,
    max_per_attack_combo: int | None,
    seed: int = 42,
):
    """keep all benign; optionally sample <=N attack rows per combo_key."""
    if max_per_attack_combo is None or max_per_attack_combo <= 0:
        return list(entries) if not isinstance(entries, list) else entries

    rng = random.Random(seed)
    benign = []
    by_combo: dict[str, list] = defaultdict(list)
    for entry in entries:
        if entry.get("variant") == "attack" and entry.get("combo_key"):
            by_combo[str(entry["combo_key"])].append(entry)
        else:
            benign.append(entry)

    out = list(benign)
    for key in sorted(by_combo):
        rows = by_combo[key]
        if len(rows) <= max_per_attack_combo:
            out.extend(rows)
        else:
            out.extend(rng.sample(rows, max_per_attack_combo))
    return out

# iterates over the filtered manifest entries
def iter_filtered_manifest_entries(
    root_dir,
    allowlist: set[str] | None,
    check_paths=False,
    max_per_attack_combo: int | None = None,
    seed: int = 42,
):
    """yield allowlist-filtered entries, optionally capped per attack combo."""
    if max_per_attack_combo is None or max_per_attack_combo <= 0:
        for entry in iter_manifest_entries(root_dir, check_paths=check_paths):
            if manifest_entry_allowed(entry, allowlist):
                yield entry
        return

    rng = random.Random(seed)
    reservoirs: dict[str, list] = {}
    seen: dict[str, int] = defaultdict(int)

    for entry in iter_manifest_entries(root_dir, check_paths=check_paths):
        if not manifest_entry_allowed(entry, allowlist):
            continue
        if entry.get("variant") == "attack" and entry.get("combo_key"):
            key = str(entry["combo_key"])
            seen[key] += 1
            n = seen[key]
            bucket = reservoirs.setdefault(key, [])
            if len(bucket) < max_per_attack_combo:
                bucket.append(entry)
            else:
                j = rng.randint(1, n)
                if j <= max_per_attack_combo:
                    bucket[j - 1] = entry
            continue
        yield entry

    for key in sorted(reservoirs):
        yield from reservoirs[key]

# counts the filtered manifest entries
def count_filtered_manifest_entries(
    root_dir,
    allowlist: set[str] | None,
    max_per_attack_combo: int | None = None,
    seed: int = 42,
) -> int:
    if allowlist is None and (max_per_attack_combo is None or max_per_attack_combo <= 0):
        return count_manifest_entries(root_dir)
    return sum(
        1
        for _ in iter_filtered_manifest_entries(
            root_dir,
            allowlist,
            max_per_attack_combo=max_per_attack_combo,
            seed=seed,
        )
    )

# loads the simulation benign items
def load_sim_benign_items(green_benign_dir, red_benign_dir, max_per_class=None):
    items = []
    benign_dirs = {"green": green_benign_dir, "red": red_benign_dir}
    for benign_color, benign_dir in benign_dirs.items():
        if not os.path.isdir(benign_dir):
            raise FileNotFoundError(f"{benign_color} benign directory not found: {benign_dir}")
        sim_images = list_image_files(benign_dir)
        if not sim_images:
            raise FileNotFoundError(f"No images found in {benign_color} benign dir: {benign_dir}")
        if max_per_class is not None:
            sim_images = sim_images[:max_per_class]
        gt_state = BENIGN_GT_STATE[benign_color]
        print(f"{benign_color} benign: {benign_dir} ({len(sim_images)} image(s), gt_state={gt_state})")
        for sim_image in sim_images:
            items.append({
                "image": sim_image,
                "path": os.path.join(benign_dir, sim_image),
                "benign_color": benign_color,
                "gt_state": gt_state,
                "source": "sim",
                "benign_dir": benign_dir,
                "frame_path": None,
                "bbox": None,
                "attributes": None,
            })
    return items

# generates a simulated dataset
def generate_simulated_dataset(
    green_benign_dir,
    red_benign_dir,
    red_laser_dir,
    green_laser_dir,
    ir_laser_dir,
    output_dir,
    max_save_benign=10,
    max_save_attack=10,
    benign_source="sim",
    dtld_root=None,
    dtld_split="test",
    dtld_labels=None,
    dtld_max_per_class=None,
    max_per_class=None,
    dtld_pad=4,
    dtld_attr_filters=None,
    dtld_exclude_unknown=True,
    realistic_only=False,
    eval_default=True,
    benign_only=False,
    return_entries=True,
):
    """generate benign images and (optionally) laser attacks."""
    import gc

    if realistic_only:
        eval_default = False

    generated_root = os.path.join(output_dir, "generated")
    samples_benign_dir = os.path.join(output_dir, "samples", "benign")
    samples_attack_dir = os.path.join(output_dir, "samples", "attack")
    dtld_crops_dir = os.path.join(generated_root, "benign_crops")
    manifest_jsonl = os.path.join(output_dir, "manifest_entries.jsonl")

    if os.path.isdir(generated_root):
        shutil.rmtree(generated_root)
    os.makedirs(generated_root, exist_ok=True)
    if os.path.isfile(manifest_jsonl):
        os.remove(manifest_jsonl)

    if benign_source == "dtld":
        if not dtld_root:
            raise ValueError("dtld_root is required when benign_source='dtld'")
        required_attrs = (
            dtld_attr_filters if dtld_attr_filters is not None else DEFAULT_ATTR_FILTERS
        )
        print(f"DTLD attribute filter: {required_attrs}")
        dtld_cap = dtld_max_per_class if dtld_max_per_class is not None else max_per_class
        benign_items, dtld_stats = load_dtld_benign_crop_items(
            dtld_root=dtld_root,
            eval_classes=DTLD_BENIGN_COLORS,
            crops_dir=dtld_crops_dir,
            split=dtld_split,
            labels_file=dtld_labels,
            max_per_class=dtld_cap,
            pad=dtld_pad,
            required_attrs=required_attrs,
            exclude_unknown=dtld_exclude_unknown,
        )
        print(f"DTLD load stats: {json.dumps(dtld_stats, indent=2)}")
    else:
        benign_items = load_sim_benign_items(
            green_benign_dir, red_benign_dir, max_per_class=max_per_class
        )

    laser_patterns = {}
    active_laser_colors = laser_colors_for_mode(
        realistic_only=realistic_only, eval_default=eval_default
    )
    laser_dir_map = {"red": red_laser_dir, "green": green_laser_dir, "ir": ir_laser_dir}

    if not benign_only:
        for laser_color in active_laser_colors:
            laser_dir = laser_dir_map[laser_color]
            if not os.path.isdir(laser_dir):
                raise FileNotFoundError(f"{laser_color} laser directory not found: {laser_dir}")
            patterns, pattern_names = build_attack_patterns(laser_dir)
            laser_patterns[laser_color] = {
                "dir": laser_dir,
                "patterns": patterns,
                "pattern_names": pattern_names,
            }
            print(
                f"{laser_color} laser: {laser_dir} "
                f"({len(patterns)} pattern(s), "
                f"ref={os.path.basename(_laser_reference_path(laser_dir))})"
            )

    benign_saved = 0
    attack_saved = 0
    skipped_attacks = 0
    combo_count = 0
    n_benign = 0
    n_attack = 0

    def _append_manifest_entry(entry, fh):
        fh.write(json.dumps(entry, separators=(",", ":"), ensure_ascii=False))
        fh.write("\n")

    with open(manifest_jsonl, "w", encoding="utf-8") as manifest_fh:
        for item in benign_items:
            saved_sample = None
            if benign_saved < max_save_benign:
                os.makedirs(samples_benign_dir, exist_ok=True)
                saved_sample = os.path.join(
                    samples_benign_dir, f"{item['benign_color']}_{item['image']}"
                )
                shutil.copy2(item["path"], saved_sample)
                benign_saved += 1

            _append_manifest_entry({
                "image": item["image"],
                "path": item["path"],
                "saved_sample": saved_sample,
                "variant": "benign",
                "benign_color": item["benign_color"],
                "benign_dir": item["benign_dir"],
                "benign_source": item["source"],
                "frame_path": item.get("frame_path"),
                "bbox": item.get("bbox"),
                "attributes": item.get("attributes"),
                "laser_color": None,
                "laser_dir": None,
                "flip_y": None,
                "target_module": None,
                "placement": None,
                "combo_key": None,
                "pattern_id": None,
                "pattern_name": None,
                "gt_state": item["gt_state"],
            }, manifest_fh)
            n_benign += 1

        if benign_only:
            print("benign_only: skipping attack synthesis")
        else:
            if realistic_only:
                print(
                    "realistic_only: green→red laser on top (off), "
                    "red→green laser on bottom (off); no IR"
                )
            elif eval_default:
                print(
                    f"eval_default: {len(DEFAULT_EVAL_ATTACK_COMBOS)} curated "
                    "attack combos (notebook EXAMPLE_ATTACK_CASES)"
                )
                for state in sorted({i["benign_color"] for i in benign_items}):
                    print(
                        f"  {state}: "
                        f"combos={attack_injections_for_mode(state, eval_default=True)}"
                    )
            else:
                print(
                    "off-module injections: red/green/ir lasers into unlit modules only "
                    f"(states={sorted({i['benign_color'] for i in benign_items})})"
                )
                for state in sorted({i["benign_color"] for i in benign_items}):
                    print(
                        f"  {state}: off={off_modules_for_state(state)} "
                        f"combos={attack_injections_for_mode(state, realistic_only=False)}"
                    )

            benign_by_color = defaultdict(list)
            for item in benign_items:
                benign_by_color[item["benign_color"]].append(item)

            # Pre-create combo/pattern dirs once.
            combo_plan = []
            for benign_color, color_items in benign_by_color.items():
                injections = attack_injections_for_mode(
                    benign_color,
                    realistic_only=realistic_only,
                    eval_default=eval_default,
                )
                if not injections:
                    print(f"  skip {benign_color}: no valid injections in this mode")
                    continue
                for laser_color, target_module in injections:
                    key = combo_key(benign_color, laser_color, target_module)
                    combo_count += 1
                    combo_out_dir = os.path.join(generated_root, key)
                    laser_info = laser_patterns[laser_color]
                    pattern_dirs = []
                    for pattern_idx, pattern_name in enumerate(
                        laser_info["pattern_names"], start=1
                    ):
                        pattern_dir = os.path.join(combo_out_dir, f"pattern_{pattern_idx}")
                        os.makedirs(pattern_dir, exist_ok=True)
                        pattern_dirs.append((pattern_idx, pattern_name, pattern_dir))
                    combo_plan.append({
                        "benign_color": benign_color,
                        "color_items": color_items,
                        "gt_state": BENIGN_GT_STATE.get(benign_color, benign_color),
                        "laser_color": laser_color,
                        "target_module": target_module,
                        "key": key,
                        "laser_info": laser_info,
                        "pattern_dirs": pattern_dirs,
                    })
                    print(
                        f"Planned combo {combo_count}: {laser_color} laser on "
                        f"{benign_color} benign → module={target_module} "
                        f"({len(color_items)} crops × {len(pattern_dirs)} patterns)"
                    )

            # Load each crop once; apply all injections/patterns for that crop.
            color_to_plans = defaultdict(list)
            for plan in combo_plan:
                color_to_plans[plan["benign_color"]].append(plan)

            for benign_color, plans in color_to_plans.items():
                color_items = plans[0]["color_items"]
                gt_state = plans[0]["gt_state"]
                print(
                    f"Synthesizing attacks for {benign_color} "
                    f"({len(color_items)} crops, {len(plans)} injection(s))..."
                )
                for item_idx, item in enumerate(color_items, start=1):
                    sim_bgr = cv2.imread(item["path"])
                    if sim_bgr is None:
                        raise ValueError(f"Could not read benign image: {item['path']}")

                    for plan in plans:
                        laser_info = plan["laser_info"]
                        laser_color = plan["laser_color"]
                        target_module = plan["target_module"]
                        key = plan["key"]
                        patterns = laser_info["patterns"]

                        for (pattern_idx, pattern_name, pattern_dir), pattern in zip(
                            plan["pattern_dirs"], patterns
                        ):
                            attacked = paste_laser_pattern(
                                sim_bgr,
                                pattern,
                                sim_threshold=SIM_BRIGHT_THRESHOLD,
                                attributes=item.get("attributes"),
                                benign_state=gt_state,
                                target_module=target_module,
                            )
                            if attacked is None:
                                skipped_attacks += 1
                                continue

                            attack_path = os.path.join(pattern_dir, item["image"])
                            cv2.imwrite(attack_path, attacked)

                            saved_sample = None
                            if attack_saved < max_save_attack:
                                os.makedirs(samples_attack_dir, exist_ok=True)
                                saved_sample = os.path.join(
                                    samples_attack_dir,
                                    f"{key}_pattern{pattern_idx}_{item['image']}",
                                )
                                shutil.copy2(attack_path, saved_sample)
                                attack_saved += 1

                            _append_manifest_entry({
                                "image": item["image"],
                                "path": attack_path,
                                "saved_sample": saved_sample,
                                "variant": "attack",
                                "benign_color": benign_color,
                                "benign_dir": item["benign_dir"],
                                "benign_source": item["source"],
                                "frame_path": item.get("frame_path"),
                                "bbox": item.get("bbox"),
                                "attributes": item.get("attributes"),
                                "laser_color": laser_color,
                                "laser_dir": laser_info["dir"],
                                "flip_y": None,
                                "target_module": target_module,
                                "placement": target_module,
                                "combo_key": key,
                                "pattern_id": pattern_idx,
                                "pattern_name": pattern_name,
                                "gt_state": gt_state,
                            }, manifest_fh)
                            n_attack += 1

                    del sim_bgr
                    if item_idx % 50 == 0 or item_idx == len(color_items):
                        print(
                            f"  {benign_color}: {item_idx}/{len(color_items)} crops "
                            f"(attack entries so far={n_attack})"
                        )
                        gc.collect()

    # Assemble final manifest.json from streamed entries (never all in RAM at once).
    manifest_path = os.path.join(output_dir, "manifest.json")
    _assemble_manifest_json(
        manifest_jsonl,
        manifest_path,
        meta={
            "benign_source": benign_source,
            "realistic_only": realistic_only,
            "benign_only": benign_only,
            "dtld_attr_filters": dtld_attr_filters if benign_source == "dtld" else None,
            "dtld_exclude_unknown": dtld_exclude_unknown if benign_source == "dtld" else None,
        },
    )
    print(f"Generated {combo_count} attack combination(s)")
    print(f"Evaluating {n_benign} benign images")
    print(f"Evaluating {n_attack} attack images")
    print(f"Saved {benign_saved} benign sample(s) to {samples_benign_dir}")
    if not benign_only:
        print(f"Saved {attack_saved} attack sample(s) to {samples_attack_dir}")
    if skipped_attacks:
        print(f"Note: {skipped_attacks} attack(s) skipped (no bright region detected)")
    print(f"Manifest saved to {manifest_path}")

    if not return_entries:
        return []

    manifest = []
    with open(manifest_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                manifest.append(json.loads(line))
    return manifest

# assembles a manifest from a jsonl file
def _assemble_manifest_json(jsonl_path, out_path, meta):
    """write compact manifest.json from streamed jsonl entries."""
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("{")
        keys = list(meta.keys())
        for i, key in enumerate(keys):
            out.write(json.dumps(key))
            out.write(":")
            out.write(json.dumps(meta[key], ensure_ascii=False))
            out.write(",")
        out.write('"entries":[')
        first = True
        with open(jsonl_path, "r", encoding="utf-8") as inp:
            for line in inp:
                line = line.strip()
                if not line:
                    continue
                if not first:
                    out.write(",")
                first = False
                out.write(line)
        out.write("]}")

# loads a manifest from a directory
def load_manifest(output_dir):
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    # Thin pointer written by memory-safe eval (entries live in synth_cache_dir).
    if isinstance(data, dict) and data.get("synth_cache_dir"):
        cache = data["synth_cache_dir"]
        print(
            f"manifest.json is a cache pointer → streaming from {cache}"
        )
        return list(iter_manifest_entries(cache, check_paths=False))
    raise ValueError(f"Unrecognized manifest format in {manifest_path}")

# loads the manifest metadata
def load_manifest_meta(output_dir):
    """load manifest metadata without requiring full entries in RAM when possible."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    try:
        size = os.path.getsize(manifest_path)
    except OSError:
        size = 0
    if size > 50 * 1024 * 1024:  # >50MB: treat as unsafe to fully parse for meta
        # Prefer sibling cache_meta / explicit synth_cache pointer sniff.
        cache_meta = os.path.join(output_dir, "cache_meta.json")
        if os.path.isfile(cache_meta):
            with open(cache_meta, "r", encoding="utf-8") as f:
                return json.load(f)
        # Sniff the start of the file for synth_cache_dir without full parse.
        with open(manifest_path, "r", encoding="utf-8") as f:
            head = f.read(4096)
        if '"synth_cache_dir"' in head:
            # Best-effort extract via regex for pointer-like files that grew somehow.
            import re
            m = re.search(r'"synth_cache_dir"\s*:\s*"([^"]+)"', head)
            if m:
                return {"synth_cache_dir": m.group(1), "n_entries": None}
        raise ValueError(
            f"manifest.json at {manifest_path} is {size / 1e9:.2f} GB. "
            "Do not json.load it. Pass --synth_cache_dir pointing at the laser_cache "
            "with manifest_entries.jsonl, or replace manifest.json with a thin pointer."
        )


    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"entries": data}
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unrecognized manifest format in {manifest_path}")

# resolves the manifest jsonl file
def resolve_manifest_jsonl(root_dir):
    """Prefer streamed jsonl over monolithic manifest.json."""
    root_dir = _abspath(root_dir)
    for name in ("manifest_entries.jsonl", "manifest.jsonl"):
        path = os.path.join(root_dir, name)
        if os.path.isfile(path):
            return path
    return None

# counts the manifest entries
def count_manifest_entries(root_dir):
    """Count entries without loading the full manifest into RAM."""
    root_dir = _abspath(root_dir)
    jsonl = resolve_manifest_jsonl(root_dir)
    if jsonl:
        n = 0
        with open(jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
        return n
    meta_path = os.path.join(root_dir, "cache_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("n_entries") is not None:
            return int(meta["n_entries"])
    return len(load_manifest(root_dir))

# absolutizes the entry paths
def _absolutize_entry_paths(entry, root):
    e = dict(entry)
    root = _abspath(root)
    for key in ("path", "saved_sample"):
        p = e.get(key)
        if not p:
            continue
        if not os.path.isabs(p):
            e[key] = os.path.normpath(os.path.join(root, p))
        else:
            e[key] = p
    return e

# iterates over the manifest entries
def iter_manifest_entries(root_dir, check_paths=False):
    """
    Yield manifest entries one at a time (memory-safe).

    Prefers manifest_entries.jsonl. Resolves relative image paths against root_dir.
    """
    root_dir = _abspath(root_dir)
    jsonl = resolve_manifest_jsonl(root_dir)
    if jsonl:
        missing = 0
        with open(jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = _absolutize_entry_paths(json.loads(line), root_dir)
                if check_paths and entry.get("path") and not os.path.isfile(entry["path"]):
                    missing += 1
                yield entry
        if check_paths and missing:
            raise FileNotFoundError(
                f"{missing} manifest image path(s) missing under {root_dir}. "
                "Re-run with --force_resynth to rebuild."
            )
        return

    entries = load_manifest(root_dir)
    for entry in _make_paths_absolute_from_root(entries, root_dir):
        yield entry

# writes an evaluation manifest pointer
def write_eval_manifest_pointer(output_dir, synth_cache_dir, n_entries, args):
    """write manifest stup"""
    os.makedirs(output_dir, exist_ok=True)
    pointer = {
        "benign_source": args.benign_source,
        "realistic_only": args.realistic_only,
        "benign_only": args.benign_only,
        "synth_cache_dir": _abspath(synth_cache_dir),
        "n_entries": int(n_entries),
        "entries_ref": "manifest_entries.jsonl",
        "note": (
            "Full attack entries live in synth_cache_dir/manifest_entries.jsonl. "
            "Eval streams from there shouldn't need to load the full manifest into RAM, i hope."
        ),
    }
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pointer, f, indent=2)
    return path

# checks if a synth cache is ready
def synth_cache_ready(cache_dir):
    if not cache_dir:
        return False
    has_manifest = (
        os.path.isfile(os.path.join(cache_dir, "manifest.json"))
        or os.path.isfile(os.path.join(cache_dir, "manifest_entries.jsonl"))
    )
    return has_manifest and os.path.isdir(os.path.join(cache_dir, "generated"))

# absolutizes a path
def _abspath(path):
    return os.path.abspath(os.path.expanduser(path)) if path else path

# makes paths relative to the root
def _make_paths_relative_to_root(entries, root):
    """rewrite image paths under root to be relative."""
    root = _abspath(root)
    out = []
    for entry in entries:
        e = dict(entry)
        for key in ("path", "saved_sample"):
            p = e.get(key)
            if not p:
                continue
            ap = _abspath(p)
            if ap.startswith(root + os.sep) or ap == root:
                e[key] = os.path.relpath(ap, root)
        out.append(e)
    return out

# makes paths absolute from the root
def _make_paths_absolute_from_root(entries, root):
    """Resolve relative image paths against the synth cache root."""
    root = _abspath(root)
    out = []
    missing = 0
    for entry in entries:
        e = dict(entry)
        for key in ("path", "saved_sample"):
            p = e.get(key)
            if not p:
                continue
            if not os.path.isabs(p):
                p = os.path.normpath(os.path.join(root, p))
            e[key] = p
            if key == "path" and not os.path.isfile(p):
                missing += 1
        out.append(e)
    if missing:
        raise FileNotFoundError(
            f"{missing} manifest image path(s) missing under synth cache {root}. "
            "Re-run with --force_resynth to rebuild."
        )
    return out

# builds a synth cache fingerprint
def build_synth_cache_fingerprint(args):
    """Settings that define the contents of a synth cache."""
    attr_filters = (
        {} if getattr(args, "no_attr_filter", False)
        else load_attr_filters(getattr(args, "attr_filter_json", None))
    )
    return {
        "benign_source": args.benign_source,
        "realistic_only": bool(args.realistic_only),
        "benign_only": bool(args.benign_only),
        "dtld_root": _abspath(args.dtld_root) if args.dtld_root else None,
        "dtld_split": args.dtld_split,
        "dtld_labels": _abspath(args.dtld_labels) if args.dtld_labels else None,
        "dtld_max_per_class": args.dtld_max_per_class,
        "max_per_class": args.max_per_class,
        "dtld_pad": args.dtld_pad,
        "dtld_attr_filters": attr_filters,
        "dtld_exclude_unknown": not args.include_unknown,
        "red_laser_dir": _abspath(args.red_laser_dir),
        "green_laser_dir": _abspath(args.green_laser_dir),
        "ir_laser_dir": _abspath(args.ir_laser_dir) if not args.realistic_only else None,
        "dtld_benign_colors": list(DTLD_BENIGN_COLORS),
        "laser_colors": list(
            laser_colors_for_mode(
                realistic_only=bool(args.realistic_only),
                eval_default=(
                    not bool(args.realistic_only)
                    and not bool(getattr(args, "all_attack_combos", False))
                ),
            )
        ),
        "all_attack_combos": bool(getattr(args, "all_attack_combos", False)),
        "eval_attack_combos": (
            None
            if bool(getattr(args, "all_attack_combos", False))
            or bool(args.realistic_only)
            else list(DEFAULT_EVAL_ATTACK_COMBOS)
        ),
    }

# saves the synth cache metadata
def save_synth_cache_meta(cache_dir, fingerprint, n_entries, n_attack, n_benign):
    meta = {
        **fingerprint,
        "n_entries": n_entries,
        "n_attack": n_attack,
        "n_benign": n_benign,
    }
    meta_path = os.path.join(cache_dir, "cache_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path

# loads a synth cache
def load_synth_cache(cache_dir, expected_fingerprint=None, return_entries=False):
    """
    Validate / summarize a synth cache.
    """
    cache_dir = _abspath(cache_dir)
    if not synth_cache_ready(cache_dir):
        raise FileNotFoundError(f"Synth cache not ready: {cache_dir}")

    meta_path = os.path.join(cache_dir, "cache_meta.json")
    if expected_fingerprint and os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        mismatched = []
        for key, expected in expected_fingerprint.items():
            if cached.get(key) != expected:
                mismatched.append(key)
        if mismatched:
            print(
                "WARNING: synth cache settings differ from current run "
                f"({', '.join(mismatched)}). Using cache anyway; "
                "pass --force_resynth to rebuild."
            )

    n_entries = count_manifest_entries(cache_dir)
    n_attack = n_benign = None
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        n_attack = meta.get("n_attack")
        n_benign = meta.get("n_benign")
        if meta.get("n_entries") is not None:
            n_entries = int(meta["n_entries"])
    if n_attack is None or n_benign is None:
        n_attack = n_benign = 0
        for entry in iter_manifest_entries(cache_dir):
            if entry.get("variant") == "attack":
                n_attack += 1
            elif entry.get("variant") == "benign":
                n_benign += 1
    print(
        f"Loaded synth cache from {cache_dir} "
        f"({n_benign} benign, {n_attack} attack, {n_entries} total) "
        f"[streaming; not held in RAM]"
    )
    if return_entries:
        return list(iter_manifest_entries(cache_dir, check_paths=True))
    return []

# makes paths relative to the root entry
def _make_paths_relative_to_root_entry(entry, root):
    e = dict(entry)
    for key in ("path", "saved_sample"):
        p = e.get(key)
        if not p:
            continue
        ap = _abspath(p)
        if ap.startswith(root + os.sep) or ap == root:
            e[key] = os.path.relpath(ap, root)
    return e

# finalizes a synth cache
def finalize_synth_cache(cache_dir, fingerprint, return_entries=False):
    """rewrite manifest paths to be cache-relative and write cache_meta.json."""
    cache_dir = _abspath(cache_dir)
    jsonl_path = os.path.join(cache_dir, "manifest_entries.jsonl")
    manifest_path = os.path.join(cache_dir, "manifest.json")

    if not os.path.isfile(jsonl_path):
        # fallback
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"No manifest_entries.jsonl or manifest.json under {cache_dir}"
            )
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            with open(manifest_path, "r", encoding="utf-8") as src:
                decoder = json.JSONDecoder()
                raw = src.read()
            # Prefer streaming export only when entries already exist as jsonl;
            # otherwise rewrite by parsing once.
            data = decoder.decode(raw)
            del raw
            if isinstance(data, list):
                entries = data
            elif isinstance(data, dict) and "entries" in data:
                entries = data["entries"]
            else:
                raise ValueError(f"Unrecognized manifest format in {manifest_path}")
            for entry in entries:
                fh.write(json.dumps(entry, separators=(",", ":"), ensure_ascii=False))
                fh.write("\n")
            del entries, data

    # stream-relative rewrite into a temp jsonl, then assemble compact manifest.json.
    rel_jsonl = os.path.join(cache_dir, "manifest_entries.rel.jsonl")
    n_attack = n_benign = 0
    with open(jsonl_path, "r", encoding="utf-8") as inp, open(
        rel_jsonl, "w", encoding="utf-8"
    ) as out:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            entry = _make_paths_relative_to_root_entry(json.loads(line), cache_dir)
            if entry.get("variant") == "attack":
                n_attack += 1
            elif entry.get("variant") == "benign":
                n_benign += 1
            out.write(json.dumps(entry, separators=(",", ":"), ensure_ascii=False))
            out.write("\n")

    os.replace(rel_jsonl, jsonl_path)
    meta = {
        "benign_source": fingerprint.get("benign_source"),
        "realistic_only": fingerprint.get("realistic_only"),
        "benign_only": fingerprint.get("benign_only"),
        "dtld_attr_filters": fingerprint.get("dtld_attr_filters"),
        "dtld_exclude_unknown": fingerprint.get("dtld_exclude_unknown"),
    }
    _assemble_manifest_json(jsonl_path, manifest_path, meta)
    meta_path = save_synth_cache_meta(
        cache_dir, fingerprint, n_benign + n_attack, n_attack, n_benign
    )
    print(f"Synth cache finalized at {cache_dir}")
    print(f"  manifest: {manifest_path}")
    print(f"  entries:  {jsonl_path}")
    print(f"  meta: {meta_path}")
    print(f"  layout: generated/<combo_key>/pattern_<id>/<image>")
    if not return_entries:
        return []
    return _make_paths_absolute_from_root(load_manifest(cache_dir), cache_dir)

def compute_combination_metrics(rows, variant="attack"):
    """
    Aggregate state attack metrics by (benign_color, laser_color, placement/module).
    """
    subset = [r for r in rows if r["variant"] == variant]
    if not subset:
        return []

    grouped = defaultdict(list)
    for row in subset:
        key = (
            row.get("benign_color", ""),
            row.get("laser_color", ""),
            row.get("placement", "") or row.get("target_module", ""),
        )
        grouped[key].append(row)

    results = []
    for (benign_color, laser_color, placement), group in grouped.items():
        total = len(group)
        attack_successes = sum(int(r["attack_success"]) for r in group)
        state_correct = sum(int(r["state_correct"]) for r in group)

        results.append({
            "benign_color": benign_color,
            "laser_color": laser_color,
            "placement": placement,
            "target_module": placement,
            "flip_y": "",
            "combo_key": combo_key(benign_color, laser_color, placement)
            if benign_color and laser_color and placement else "",
            "total": total,
            "attack_success_rate": attack_successes / total,
            "state_error_rate": attack_successes / total,
            "state_accuracy": state_correct / total,
            "attack_successes": attack_successes,
            "state_correct": state_correct,
        })

    results.sort(key=lambda r: (-r["attack_success_rate"], -r["total"]))
    return results

# computes the pattern combination metrics
def compute_pattern_combination_metrics(rows, variant="attack"):
    """Same as compute_combination_metrics but also splits by pattern_id."""
    subset = [r for r in rows if r["variant"] == variant]
    if not subset:
        return []

    grouped = defaultdict(list)
    for row in subset:
        key = (
            row.get("benign_color", ""),
            row.get("laser_color", ""),
            row.get("placement", "") or row.get("target_module", ""),
            row.get("pattern_id", ""),
        )
        grouped[key].append(row)

    results = []
    for (benign_color, laser_color, placement, pattern_id), group in grouped.items():
        total = len(group)
        attack_successes = sum(int(r["attack_success"]) for r in group)
        state_correct = sum(int(r["state_correct"]) for r in group)

        results.append({
            "benign_color": benign_color,
            "laser_color": laser_color,
            "placement": placement,
            "target_module": placement,
            "flip_y": "",
            "pattern_id": pattern_id,
            "combo_key": combo_key(benign_color, laser_color, placement)
            if benign_color and laser_color and placement else "",
            "total": total,
            "attack_success_rate": attack_successes / total,
            "state_error_rate": attack_successes / total,
            "state_accuracy": state_correct / total,
            "attack_successes": attack_successes,
            "state_correct": state_correct,
        })

    results.sort(key=lambda r: (-r["attack_success_rate"], -r["total"]))
    return results

# converts a value to a boolean
def _csv_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in ("", "nan", "none"):
        return False
    return s in ("1", "true", "yes")

# rebuilds a summary from a results csv
def rebuild_summary_from_results_csv(csv_path, tag="BASELINE", model_path=""):
    """
    Stream a results_*.csv and rebuild attack rankings + headline metrics.

    Used when a model finished writing its CSV but did not write summary.json / attack_ranking.json.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    combo_stats = defaultdict(lambda: {"total": 0, "attack_success": 0, "state_correct": 0})
    pattern_stats = defaultdict(lambda: {"total": 0, "attack_success": 0, "state_correct": 0})
    n_total = n_attack = n_attack_success = 0
    state_correct = 0
    benign_correct = benign_total = 0
    # metric_rows kept compact for sklearn state metrics
    metric_rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            variant = row.get("variant", "")
            gt_state = row.get("gt_state", "")
            pred_state = row.get("pred_state", "")
            sc = _csv_bool(row.get("state_correct"))
            if sc:
                state_correct += 1

            metric_rows.append({
                "variant": variant,
                "gt_state": gt_state,
                "pred_state": pred_state,
            })

            if variant == "benign":
                benign_total += 1
                if sc:
                    benign_correct += 1
                continue

            if variant != "attack":
                continue

            n_attack += 1
            atk = _csv_bool(row.get("attack_success"))
            # Fall back: attack success = state mismatch
            if row.get("attack_success", "") == "" and gt_state and pred_state:
                atk = pred_state != gt_state
            if atk:
                n_attack_success += 1

            benign_color = row.get("benign_color", "") or ""
            laser_color = row.get("laser_color", "") or ""
            placement = row.get("placement", "") or row.get("target_module", "") or ""
            pattern_id = row.get("pattern_id", "") or ""
            ck = (benign_color, laser_color, placement)
            combo_stats[ck]["total"] += 1
            combo_stats[ck]["state_correct"] += int(sc)
            combo_stats[ck]["attack_success"] += int(atk)
            pk = (benign_color, laser_color, placement, pattern_id)
            pattern_stats[pk]["total"] += 1
            pattern_stats[pk]["state_correct"] += int(sc)
            pattern_stats[pk]["attack_success"] += int(atk)

    attack_ranking = _finalize_combo_stats(combo_stats, with_pattern=False)
    pattern_attack_ranking = _finalize_combo_stats(pattern_stats, with_pattern=True)
    attack_rate = n_attack_success / n_attack if n_attack else 0.0
    state_acc = state_correct / n_total if n_total else 0.0
    benign_state_acc = benign_correct / benign_total if benign_total else 0.0

    state_metrics = compute_state_metrics(metric_rows)
    benign_state_metrics = compute_state_metrics(metric_rows, "benign")
    attack_state_metrics = compute_state_metrics(metric_rows, "attack")
    del metric_rows

    print(
        f"Rebuilt {tag} from {csv_path}: n={n_total}, "
        f"attack_success={attack_rate * 100:.2f}% ({n_attack_success}/{n_attack})"
    )

    return {
        "model": os.path.basename(model_path) if model_path else os.path.basename(csv_path),
        "model_path": model_path,
        "tag": tag,
        "total": n_total,
        "state_accuracy": state_acc,
        "state_correct": state_correct,
        "state_total": n_total,
        "attack_state_success_rate": attack_rate,
        "benign_state_accuracy": benign_state_acc,
        "benign_state_metrics": benign_state_metrics,
        "attack_state_metrics": attack_state_metrics,
        "state_metrics": state_metrics,
        "attack_ranking": attack_ranking,
        "pattern_attack_ranking": pattern_attack_ranking,
        "most_effective_attack": attack_ranking[0] if attack_ranking else None,
        "least_effective_attack": attack_ranking[-1] if attack_ranking else None,
        "csv_path": csv_path,
        "rebuilt_from_csv": True,
    }

# prints the attack ranking
def print_attack_ranking(tag, combination_metrics, pattern_metrics):
    print(f"\n--- {tag} Attack effectiveness ranking (by state misclassification) ---")
    if not combination_metrics:
        print("  no attack combination data")
        return

    print("Most effective → least effective (higher = model fooled more often):")
    for i, combo in enumerate(combination_metrics, start=1):
        print(
            f"  {i}. {combo['laser_color']} laser on {combo['benign_color']} "
            f"({placement_display(combo['placement'])}): "
            f"attack_success={combo['attack_success_rate'] * 100:.1f}% "
            f"state_acc={combo['state_accuracy'] * 100:.1f}% (n={combo['total']})"
        )

    if pattern_metrics:
        best = pattern_metrics[0]
        print(
            f"\nMost effective single pattern: {best['laser_color']} laser on "
            f"{best['benign_color']} ({placement_display(best['placement'])}) pattern={best['pattern_id']} "
            f"(attack_success={best['attack_success_rate'] * 100:.1f}%)"
        )

# prints the cross-model attack ranking
def print_cross_model_attack_ranking(summaries):
    print(f"\n{'=' * 60}")
    print("MOST EFFECTIVE ATTACKS BY MODEL (state misclassification)")
    print(f"{'=' * 60}")

    preferred = ("baseline", "ontology", "ontology_2")
    model_tags = [t for t in preferred if t in summaries]
    model_tags.extend(t for t in summaries if t not in model_tags)
    for model_tag in model_tags:
        metrics = summaries.get(model_tag, {}).get("attack_ranking", [])
        if metrics:
            best = metrics[0]
            print(
                f"{model_tag}: {best['laser_color']} laser on {best['benign_color']} "
                f"({placement_display(best['placement'])}) — "
                f"attack_success={best['attack_success_rate'] * 100:.1f}%"
            )

# converts a value to a boolean
def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)

# builds a row lookup key
def _row_lookup_key(row):
    return (
        row["image"],
        row.get("combo_key", ""),
        str(row.get("pattern_id", "")),
        row.get("variant", ""),
    )

# builds a vlm payload
def _build_vlm_payload(row, pred_raw=None):
    """Format one model's raw output for a sample sidecar JSON."""
    payload = {
        "pred_state": row["pred_state"],
        "state_conf": float(row["state_conf"]),
        "state_correct": _as_bool(row["state_correct"]),
        "dom": row["dom"],
        "bright": float(row["bright"]),
        "cie_x": float(row["cie_x"]),
        "cie_bucket": row["cie_bucket"],
    }
    if row.get("attack_success") != "":
        payload["attack_success"] = _as_bool(row["attack_success"])
    if pred_raw:
        payload["state_probs"] = pred_raw.get("state_probs", {})
        payload["logits"] = pred_raw.get("logits", [])
        payload["multi_spot_brightness"] = pred_raw.get("multi", {})
        if pred_raw.get("module_analysis"):
            payload["module_analysis"] = pred_raw["module_analysis"]
    else:
        state_probs = row.get("state_probs", {})
        if isinstance(state_probs, str):
            state_probs = _parse_row_field(state_probs)
        logits = row.get("logits", [])
        if isinstance(logits, str):
            logits = _parse_row_field(logits)
        if state_probs:
            payload["state_probs"] = state_probs
        if logits:
            payload["logits"] = logits
    return payload

# evenly spaces items
def _evenly_spaced(items, count):
    if not items:
        return []
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    return [items[int(round(i * step))] for i in range(count)]

# computes the pattern ranking by combo
def compute_pattern_ranking_by_combo(model_rows, ranking_model="baseline"):
    """
    Rank laser patterns within each combo by attack_success_rate on state prediction.

    Uses one model's rows (default: baseline) to pick most/least effective patterns.
    """
    if ranking_model not in model_rows:
        ranking_model = next(iter(model_rows))

    rows = [r for r in model_rows[ranking_model] if r.get("variant") == "attack"]
    grouped = defaultdict(list)
    for row in rows:
        combo_key = row.get("combo_key")
        pattern_id = row.get("pattern_id")
        if not combo_key or pattern_id == "":
            continue
        grouped[(combo_key, int(pattern_id))].append(row)

    by_combo = defaultdict(list)
    for (combo_key, pattern_id), group in grouped.items():
        total = len(group)
        successes = sum(_as_bool(r["attack_success"]) for r in group)
        by_combo[combo_key].append({
            "pattern_id": pattern_id,
            "attack_success_rate": successes / total if total else 0.0,
            "total": total,
        })

    rankings = {}
    for combo_key, patterns in by_combo.items():
        patterns.sort(key=lambda p: (-p["attack_success_rate"], p["pattern_id"]))
        rankings[combo_key] = {
            "most_effective": patterns[0],
            "least_effective": patterns[-1],
            "all_patterns": patterns,
        }
    return rankings

# selects attack samples
def select_attack_samples(manifest, model_rows, samples_per_pattern=3, ranking_model="baseline"):
    """
    For each attack combo case, pick samples from the most and least effective patterns.

    Effectiveness is measured by state misclassification rate (attack_success) using
    the baseline model unless ranking_model is overridden.
    """
    pattern_rankings = compute_pattern_ranking_by_combo(model_rows, ranking_model)

    manifest_by_combo_pattern = defaultdict(list)
    for entry in manifest:
        if entry.get("variant") != "attack" or not entry.get("combo_key"):
            continue
        manifest_by_combo_pattern[(entry["combo_key"], entry.get("pattern_id"))].append(entry)

    selected = {}
    for combo_key in sorted(pattern_rankings):
        ranks = pattern_rankings[combo_key]
        selected[combo_key] = {
            "ranking": ranks,
            "samples": [],
        }

        tiers = [
            ("most_effective", ranks["most_effective"]),
            ("least_effective", ranks["least_effective"]),
        ]
        seen_tiers = set()
        for tier, pattern_info in tiers:
            pattern_id = pattern_info["pattern_id"]
            tier_key = (tier, pattern_id)
            if tier_key in seen_tiers:
                continue
            seen_tiers.add(tier_key)

            entries = sorted(
                manifest_by_combo_pattern.get((combo_key, pattern_id), []),
                key=lambda e: e["image"],
            )
            for entry in _evenly_spaced(entries, samples_per_pattern):
                selected[combo_key]["samples"].append({
                    "entry": entry,
                    "tier": tier,
                    "pattern_id": pattern_id,
                    "pattern_attack_success_rate": pattern_info["attack_success_rate"],
                })

    return selected

# picks a benign example
def _pick_benign_example(entries, row_map, want_correct):
    """Pick highest-confidence correct or incorrect benign prediction."""
    candidates = []
    for entry in entries:
        lookup_key = _row_lookup_key({
            "image": entry["image"],
            "combo_key": "",
            "pattern_id": "",
            "variant": "benign",
        })
        row = row_map.get(lookup_key)
        if not row:
            continue
        if _as_bool(row["state_correct"]) == want_correct:
            candidates.append((float(row["state_conf"]), entry, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: -item[0])
    conf, entry, row = candidates[0]
    return {
        "entry": entry,
        "row": row,
        "state_conf": conf,
        "tier": "correct" if want_correct else "incorrect",
    }

# selects benign samples
def select_benign_samples(manifest, model_rows, ranking_model="baseline"):
    """
    Pick one correct and one incorrect benign prediction per class (green, red).

    Uses ranking_model (default baseline) to decide correctness.
    """
    if ranking_model not in model_rows:
        ranking_model = next(iter(model_rows))

    row_map = {
        _row_lookup_key(row): row
        for row in model_rows[ranking_model]
        if row.get("variant") == "benign"
    }

    selected = {}
    benign_colors = sorted({
        e.get("benign_color")
        for e in manifest
        if e.get("variant") == "benign" and e.get("benign_color")
    }) or list(BENIGN_COLORS)
    for benign_color in benign_colors:
        entries = [
            e for e in manifest
            if e.get("variant") == "benign" and e.get("benign_color") == benign_color
        ]
        selected[benign_color] = {
            "correct": _pick_benign_example(entries, row_map, want_correct=True),
            "incorrect": _pick_benign_example(entries, row_map, want_correct=False),
        }
    return selected


def _write_vlm_sample_dir(
    samples_root,
    sample_dir_relative,
    entry,
    tier,
    model_labels,
    row_maps,
    extra_fields=None,
):
    """Write one image + vlm_output.json sidecar; return index record."""
    sample_dir = os.path.join(samples_root, sample_dir_relative)
    os.makedirs(sample_dir, exist_ok=True)

    dst_image = os.path.join(sample_dir, entry["image"])
    shutil.copy2(entry["path"], dst_image)

    lookup_key = _row_lookup_key({
        "image": entry["image"],
        "combo_key": entry.get("combo_key") or "",
        "pattern_id": entry.get("pattern_id") or "",
        "variant": entry.get("variant", ""),
    })

    record = {
        "sample_id": sample_dir_relative.replace(os.sep, "/"),
        "image": entry["image"],
        "source_path": entry["path"],
        "saved_image": dst_image,
        "variant": entry.get("variant"),
        "combo_key": entry.get("combo_key"),
        "benign_color": entry.get("benign_color"),
        "laser_color": entry.get("laser_color"),
        "placement": entry.get("placement"),
        "target_module": entry.get("target_module") or entry.get("placement"),
        "flip_y": entry.get("flip_y"),
        "pattern_id": entry.get("pattern_id"),
        "pattern_name": entry.get("pattern_name"),
        "effectiveness_tier": tier,
        "gt_state": entry.get("gt_state"),
        "models": {},
    }
    if extra_fields:
        record.update(extra_fields)

    for label in model_labels:
        row = row_maps[label].get(lookup_key)
        if row:
            record["models"][label] = _build_vlm_payload(row)

    sidecar_path = os.path.join(sample_dir, "vlm_output.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return {
        "sample_id": record["sample_id"],
        "variant": entry.get("variant"),
        "benign_color": entry.get("benign_color"),
        "combo_key": entry.get("combo_key"),
        "effectiveness_tier": tier,
        "pattern_id": entry.get("pattern_id"),
        "pattern_attack_success_rate": extra_fields.get("pattern_attack_success_rate")
        if extra_fields else None,
        "image": entry["image"],
        "gt_state": entry.get("gt_state"),
        "saved_image": dst_image,
        "vlm_output": sidecar_path,
        "baseline_pred": record["models"].get("baseline", {}).get("pred_state"),
        "ontology_pred": record["models"].get("ontology", {}).get("pred_state"),
        "ontology_2_pred": record["models"].get("ontology_2", {}).get("pred_state"),
        "state_correct": record["models"].get("baseline", {}).get("state_correct"),
    }

# saves samples of each attack and benign
def save_vlm_output_samples(
    manifest,
    model_rows,
    output_dir,
    samples_per_pattern=3,
    ranking_model="baseline",
    model_labels=None,
):
    """
    Save attack and benign VLM output samples.

    Writes:
      vlm_samples/<combo_key>/most_effective_pattern_<n>/<idx>_<image_stem>/
      vlm_samples/benign/<color>/correct|<incorrect>/<image_stem>/
        image.jpg
        vlm_output.json
      vlm_samples/index.json
    """
    if model_labels is None:
        model_labels = list(model_rows.keys())

    samples_root = os.path.join(output_dir, "vlm_samples")
    if os.path.isdir(samples_root):
        shutil.rmtree(samples_root)
    os.makedirs(samples_root, exist_ok=True)

    selected = select_attack_samples(
        manifest,
        model_rows,
        samples_per_pattern=samples_per_pattern,
        ranking_model=ranking_model,
    )
    benign_selected = select_benign_samples(
        manifest, model_rows, ranking_model=ranking_model,
    )
    index = []
    benign_index = {}

    row_maps = {}
    for label in model_labels:
        row_maps[label] = {_row_lookup_key(row): row for row in model_rows[label]}

    for combo_key in sorted(selected):
        combo_info = selected[combo_key]
        tier_counts = defaultdict(int)
        for sample in combo_info["samples"]:
            entry = sample["entry"]
            tier = sample["tier"]
            pattern_id = sample["pattern_id"]
            tier_counts[tier] += 1
            idx = tier_counts[tier]

            image_stem = os.path.splitext(entry["image"])[0]
            sample_dir_relative = (
                f"{combo_key}/{tier}_pattern_{pattern_id}/{idx:02d}_{image_stem}"
            )
            index.append(_write_vlm_sample_dir(
                samples_root,
                sample_dir_relative,
                entry,
                tier,
                model_labels,
                row_maps,
                extra_fields={
                    "pattern_attack_success_rate": sample["pattern_attack_success_rate"],
                },
            ))

    for benign_color in sorted(benign_selected.keys()):
        color_info = benign_selected[benign_color]
        benign_index[benign_color] = {}
        for tier in ("correct", "incorrect"):
            pick = color_info.get(tier)
            if not pick:
                print(f"  benign/{benign_color}: no {tier} prediction sample available")
                continue
            entry = pick["entry"]
            image_stem = os.path.splitext(entry["image"])[0]
            sample_dir_relative = f"benign/{benign_color}/{tier}/{image_stem}"
            benign_index[benign_color][tier] = _write_vlm_sample_dir(
                samples_root,
                sample_dir_relative,
                entry,
                tier,
                model_labels,
                row_maps,
            )
            index.append(benign_index[benign_color][tier])

    combo_summaries = {}
    for combo_key, combo_info in selected.items():
        ranks = combo_info["ranking"]
        combo_summaries[combo_key] = {
            "most_effective_pattern": ranks["most_effective"],
            "least_effective_pattern": ranks["least_effective"],
        }

    index_path = os.path.join(samples_root, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "samples_per_pattern": samples_per_pattern,
            "ranking_model": ranking_model,
            "cases": sorted(selected.keys()),
            "combo_summaries": combo_summaries,
            "benign_samples": benign_index,
            "total_samples": len(index),
            "samples": index,
        }, f, indent=2)

    print(f"\nSaved {len(index)} VLM output sample(s) to {samples_root}")
    for benign_color, tiers in benign_index.items():
        saved = ", ".join(sorted(tiers.keys())) or "none"
        print(f"  benign/{benign_color}: {saved}")
    for combo_key, combo_info in sorted(selected.items()):
        ranks = combo_info["ranking"]
        most = ranks["most_effective"]
        least = ranks["least_effective"]
        print(
            f"  {combo_key}: {len(combo_info['samples'])} sample(s) | "
            f"best=pattern_{most['pattern_id']} ({most['attack_success_rate'] * 100:.1f}%), "
            f"worst=pattern_{least['pattern_id']} ({least['attack_success_rate'] * 100:.1f}%)"
        )
    print(f"Sample index: {index_path}")
    return index_path

# parses a row field
def _parse_row_field(value):
    if isinstance(value, str) and value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, str) and value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value

# extracts vlm samples from a csv
def extract_vlm_samples_from_csv(output_dir, manifest, samples_per_pattern=3):
    """Rebuild VLM sample pack from existing results CSVs (no re-inference)."""
    model_rows = {}
    csv_specs = (
        ("baseline", "results_baseline.csv"),
        ("ontology", "results_ontology.csv"),
        ("ontology_2", "results_ontology_2.csv"),
    )
    for label, csv_name in csv_specs:
        csv_path = os.path.join(output_dir, csv_name)
        if not os.path.isfile(csv_path):
            if label == "ontology_2":
                continue  # optional second ontology model
            raise FileNotFoundError(f"Missing results file: {csv_path}")
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = []
            for row in csv.DictReader(f):
                row["state_probs"] = _parse_row_field(row.get("state_probs", ""))
                row["logits"] = _parse_row_field(row.get("logits", ""))
                rows.append(row)
            model_rows[label] = rows

    return save_vlm_output_samples(
        manifest,
        model_rows,
        output_dir,
        samples_per_pattern=samples_per_pattern,
    )


# eval

# formats a duration
def _format_duration(seconds):
    """Human-readable duration for progress / ETA lines."""
    if seconds is None or seconds < 0 or seconds != seconds:  # NaN check
        return "?"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"

# prints the eval timing summary
def _print_eval_timing_summary(tag, timing_totals, n_images, wall_elapsed):
    """Report where per-image eval time was spent."""
    stages = (
        ("image_load", "Image loading"),
        ("inference", "VLM inference"),
        ("color_features", "Color features"),
        ("module_analysis", "Module analysis"),
        ("postprocess", "Checker / bookkeeping"),
    )
    predict_total = sum(timing_totals.get(k, 0.0) for k, _ in stages)
    print(f"\n--- {tag} Timing breakdown ({n_images} images, wall={_format_duration(wall_elapsed)}) ---")
    if n_images <= 0 or predict_total <= 0:
        print("  (no timing data)")
        return

    ranked = sorted(
        ((k, label, timing_totals.get(k, 0.0)) for k, label in stages),
        key=lambda x: -x[2],
    )
    for _, label, secs in ranked:
        pct = 100.0 * secs / predict_total
        avg_ms = 1000.0 * secs / n_images
        print(f"  {label:<22} {secs:8.1f}s  ({pct:5.1f}%)  avg {avg_ms:6.1f} ms/img")
    print(
        f"  {'Accounted total':<22} {predict_total:8.1f}s  "
        f"({100.0 * predict_total / wall_elapsed:5.1f}% of wall)"
        if wall_elapsed > 0 else f"  {'Accounted total':<22} {predict_total:8.1f}s"
    )
    bottleneck_label = ranked[0][1]
    bottleneck_pct = 100.0 * ranked[0][2] / predict_total
    print(f"  Bottleneck: {bottleneck_label} ({bottleneck_pct:.1f}% of per-image work)")

# expected on modules for a state
def expected_on_modules_for_state(state):
    mapping = {
        "red": ("top",),
        "yellow": ("middle",),
        "green": ("bottom",),
        "red_yellow": ("top", "middle"),
        "off": (),
    }
    return mapping.get(state, ())

# prints the eval progress
def _print_eval_progress(tag, idx, total, eval_t0, timing_totals):
    """Print ETA + per-stage %; return current timestamp."""
    now = time.perf_counter()
    elapsed = now - eval_t0
    rate = idx / elapsed if elapsed > 0 else 0.0
    remaining = (total - idx) / rate if rate > 0 else 0.0
    stage_so_far = sum(timing_totals.values())
    stage_keys = (
        "image_load",
        "inference",
        "color_features",
        "module_analysis",
        "postprocess",
    )
    if stage_so_far > 0:
        stage_pcts = " ".join(
            f"{k}={100.0 * timing_totals[k] / stage_so_far:.0f}%"
            for k in stage_keys
        )
    else:
        stage_pcts = " ".join(f"{k}=0%" for k in stage_keys)
    print(
        f"  [progress {tag}] {idx}/{total} "
        f"({100.0 * idx / total:.1f}%) | "
        f"elapsed={_format_duration(elapsed)} | "
        f"ETA={_format_duration(remaining)} | "
        f"{rate:.2f} img/s | "
        f"{stage_pcts}"
    )
    return now


# csv result fields
CSV_RESULT_FIELDS = [
    "model_tag", "image", "path", "variant", "benign_color", "laser_color",
    "placement", "target_module", "flip_y", "combo_key", "pattern_id", "pattern_name",
    "gt_state", "pred_state", "state_conf",
    "state_probs", "logits", "state_correct", "expected_on_modules", "pred_on_modules",
    "module_on_correct", "pred_num_on_modules", "module_analysis_method", "scene_brightness",
    "attack_success", "dom", "bright", "cie_x", "cie_bucket",
    "primary_head", "second_head", "third_head",
]

# iterates over entry batches
def _iter_entry_batches(entry_iter, batch_size):
    batch = []
    for entry in entry_iter:
        batch.append(entry)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# finalizes combo stats
def _finalize_combo_stats(combo_stats, with_pattern=False):
    results = []
    for key, stats in combo_stats.items():
        total = stats["total"]
        if total <= 0:
            continue
        if with_pattern:
            benign_color, laser_color, placement, pattern_id = key
        else:
            benign_color, laser_color, placement = key
            pattern_id = None
        attack_successes = stats["attack_success"]
        state_correct = stats["state_correct"]
        row = {
            "benign_color": benign_color,
            "laser_color": laser_color,
            "placement": placement,
            "target_module": placement,
            "flip_y": "",
            "combo_key": combo_key(benign_color, laser_color, placement)
            if benign_color and laser_color and placement else "",
            "total": total,
            "attack_success_rate": attack_successes / total,
            "state_error_rate": attack_successes / total,
            "state_accuracy": state_correct / total,
            "attack_successes": attack_successes,
            "state_correct": state_correct,
        }
        if with_pattern:
            row["pattern_id"] = pattern_id
        results.append(row)
    results.sort(key=lambda r: (-r["attack_success_rate"], -r["total"]))
    return results

# evaluates a manifest
def evaluate_manifest(
    manifest_source,
    model_path,
    tag,
    csv_path,
    module_analysis_method=DEFAULT_MODULE_ANALYSIS_METHOD,
    progress_every=5000,
    batch_size=32,
    prefetch=True,
    prefetch_workers=2,
    total=None,
    omit_heavy_fields=True,
    keep_rows=False,
):
    """
    Evaluate a model over a manifest.
    """
    skip_module_analysis = module_analysis_method is None
    print(f"\n{'=' * 60}")
    print(f"Evaluating {tag} model: {os.path.basename(model_path)}")
    print(
        f"Module analysis: {'skipped' if skip_module_analysis else module_analysis_method}"
    )
    print(f"Batch size: {batch_size}")
    print(f"Prefetch next batch: {'on' if prefetch else 'off'}")
    print(f"CSV heavy fields (logits/probs): {'omit' if omit_heavy_fields else 'write'}")
    print(f"{'=' * 60}")

    t_load_model = time.perf_counter()
    model, classifier, preprocess, num_classes = load_model_and_classifier(model_path)
    model_load_s = time.perf_counter() - t_load_model
    print(f"Loaded model with {num_classes} classes: {CANON_STATES[:num_classes]}")
    print(f"Model load time: {_format_duration(model_load_s)}")

    if total is None:
        if hasattr(manifest_source, "__len__"):
            total = len(manifest_source)
        else:
            raise ValueError("total= is required when manifest_source has no len()")

    state_correct = state_total = 0
    module_on_correct = module_on_total = 0
    n_attack = 0
    n_attack_success = 0
    scene_brightness_cache: dict = {}

    # Compact online aggregates (never store full prediction dicts).
    combo_stats = defaultdict(lambda: {"total": 0, "attack_success": 0, "state_correct": 0})
    pattern_stats = defaultdict(lambda: {"total": 0, "attack_success": 0, "state_correct": 0})
    by_variant = defaultdict(lambda: {"attack_success": 0, "state_correct": 0, "total": 0})
    # Minimal rows for sklearn state metrics (~few hundred MB max).
    metric_rows = []
    kept_rows = [] if keep_rows else None

    timing_totals = {
        "image_load": 0.0,
        "inference": 0.0,
        "color_features": 0.0,
        "module_analysis": 0.0,
        "postprocess": 0.0,
    }
    eval_t0 = time.perf_counter()
    last_progress_t = eval_t0
    batch_size = max(1, int(batch_size))
    processed = 0

    csv_fields = [f for f in CSV_RESULT_FIELDS if not (
        omit_heavy_fields and f in ("state_probs", "logits")
    )]
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fields, extrasaction="ignore")
    csv_writer.writeheader()

    # runs a batch of predictions
    def _run_batch(batch_entries, preloaded=None, image_load_s=None):
        paths = [entry["path"] for entry in batch_entries]
        return predict_batch(
            model,
            classifier,
            preprocess,
            paths,
            num_classes,
            module_analysis_method=module_analysis_method,
            frame_paths=[entry.get("frame_path") for entry in batch_entries],
            bboxes=[entry.get("bbox") for entry in batch_entries],
            scene_brightness_cache=scene_brightness_cache,
            timings_out=timing_totals,
            preloaded=preloaded,
            image_load_s=image_load_s,
        )
    # processes predictions
    def _process_predictions(batch_entries, preds, batch_start):
        nonlocal state_correct, state_total
        nonlocal module_on_correct, module_on_total, last_progress_t
        nonlocal processed, n_attack, n_attack_success

        for offset, (entry, pred) in enumerate(zip(batch_entries, preds)):
            idx = batch_start + offset + 1
            processed = idx
            t_post0 = time.perf_counter()

            gt_state = entry["gt_state"]
            state_match = pred["state"] == gt_state
            expected_modules = tuple(expected_on_modules_for_state(gt_state))
            module_info = pred.get("module_analysis") or {}
            pred_modules = tuple(module_info.get("on_modules", []))
            if skip_module_analysis:
                module_match = False
            else:
                module_match = (
                    tuple(sorted(pred_modules)) == tuple(sorted(expected_modules))
                )
            is_attack = entry["variant"] == "attack"
            attack_success = is_attack and pred["state"] != gt_state

            state_total += 1
            if state_match:
                state_correct += 1
            if not skip_module_analysis:
                module_on_total += 1
                if module_match:
                    module_on_correct += 1
            if is_attack:
                n_attack += 1
                if attack_success:
                    n_attack_success += 1

            benign_color = entry.get("benign_color", "")
            laser_color = entry.get("laser_color") or ""
            placement = entry.get("placement") or entry.get("target_module") or ""
            pattern_id = entry.get("pattern_id") or ""
            ckey = entry.get("combo_key") or ""

            if is_attack:
                ck = (benign_color, laser_color, placement)
                combo_stats[ck]["total"] += 1
                combo_stats[ck]["state_correct"] += int(state_match)
                combo_stats[ck]["attack_success"] += int(attack_success)
                pk = (benign_color, laser_color, placement, pattern_id)
                pattern_stats[pk]["total"] += 1
                pattern_stats[pk]["state_correct"] += int(state_match)
                pattern_stats[pk]["attack_success"] += int(attack_success)

            if entry["variant"] == "benign":
                vkey = f"benign_{benign_color}"
            else:
                vkey = ckey or "attack"
                if pattern_id != "":
                    vkey += f"_p{pattern_id}"
            by_variant[vkey]["total"] += 1
            by_variant[vkey]["state_correct"] += int(state_match)
            if is_attack:
                by_variant[vkey]["attack_success"] += int(attack_success)

            metric_rows.append({
                "variant": entry["variant"],
                "gt_state": gt_state,
                "pred_state": pred["state"],
                "state_correct": state_match,
                "attack_success": int(attack_success) if is_attack else "",
                "benign_color": benign_color,
                "laser_color": laser_color,
                "placement": placement,
                "combo_key": ckey,
                "pattern_id": pattern_id,
            })

            csv_row = {
                "model_tag": tag,
                "image": entry["image"],
                "path": entry["path"],
                "variant": entry["variant"],
                "benign_color": benign_color,
                "laser_color": laser_color,
                "placement": placement,
                "target_module": entry.get("target_module") or placement,
                "flip_y": entry.get("flip_y", ""),
                "combo_key": ckey,
                "pattern_id": pattern_id,
                "pattern_name": entry.get("pattern_name") or "",
                "gt_state": gt_state,
                "pred_state": pred["state"],
                "state_conf": pred["conf"],
                "state_correct": state_match,
                "expected_on_modules": json.dumps(list(expected_modules)),
                "pred_on_modules": json.dumps(list(pred_modules)),
                "module_on_correct": (
                    "" if skip_module_analysis else module_match
                ),
                "pred_num_on_modules": (
                    "" if skip_module_analysis
                    else int(module_info.get("num_on_modules", 0))
                ),
                "module_analysis_method": (
                    "skipped" if skip_module_analysis else module_analysis_method
                ),
                "scene_brightness": module_info.get("scene_brightness", ""),
                "attack_success": int(attack_success) if is_attack else "",
                "dom": pred["dom"],
                "bright": pred["bright"],
                "cie_x": pred["cie_x"],
                "cie_bucket": pred["cie_bucket"],
                "primary_head": pred["multi"]["primary"],
                "second_head": pred["multi"]["second"],
                "third_head": pred["multi"]["third"],
            }
            if not omit_heavy_fields:
                csv_row["state_probs"] = json.dumps(pred["state_probs"])
                csv_row["logits"] = json.dumps(pred["logits"])
            csv_writer.writerow(csv_row)

            if keep_rows:
                kept_rows.append(csv_row)

            if idx % progress_every == 0 or idx == total or total <= 50:
                label = entry["variant"]
                if ckey:
                    label += f"_{ckey}"
                if pattern_id != "":
                    label += f"_p{pattern_id}"
                module_part = (
                    "modules=SKIP"
                    if skip_module_analysis
                    else (
                        f"modules={'OK' if module_match else 'MISS'} "
                        f"(pred={list(pred_modules)}, exp={list(expected_modules)})"
                    )
                )
                print(
                    f"  {label}/{entry['image']}: "
                    f"gt_state={gt_state}, pred_state={pred['state']} ({pred['conf']:.3f}) "
                    f"{'OK' if state_match else 'MISS'} | "
                    f"{module_part}"
                    + (
                        f" | attack_success={'YES' if attack_success else 'NO'}"
                        if is_attack else ""
                    )
                )

            timing_totals["postprocess"] += time.perf_counter() - t_post0
            now = time.perf_counter()
            if (
                idx == total
                or idx % progress_every == 0
                or (now - last_progress_t) >= 30.0
            ):
                last_progress_t = _print_eval_progress(
                    tag, idx, total, eval_t0, timing_totals
                )
    # tries to process predictions
    try:
        batch_iter = _iter_entry_batches(iter(manifest_source), batch_size)
        if prefetch and total > 0:
            with BatchImagePrefetcher(max_workers=max(1, int(prefetch_workers))) as prefetcher:
                try:
                    current = next(batch_iter)
                except StopIteration:
                    current = None
                if current is not None:
                    prefetcher.submit([e["path"] for e in current])
                batch_start = 0
                while current is not None:
                    try:
                        nxt = next(batch_iter)
                    except StopIteration:
                        nxt = None
                    pils, rgb_arrs, load_s = prefetcher.get()
                    if nxt is not None:
                        prefetcher.submit([e["path"] for e in nxt])
                    preds = _run_batch(
                        current,
                        preloaded=(pils, rgb_arrs),
                        image_load_s=load_s,
                    )
                    _process_predictions(current, preds, batch_start)
                    batch_start += len(current)
                    current = nxt
                    del preds, pils, rgb_arrs
        else:
            batch_start = 0
            for batch_entries in batch_iter:
                preds = _run_batch(batch_entries)
                _process_predictions(batch_entries, preds, batch_start)
                batch_start += len(batch_entries)
                del preds
    finally:
        csv_fh.flush()
        csv_fh.close()

    # Free GPU model before metric aggregation / next model.
    del model, classifier, preprocess, scene_brightness_cache
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # prints the eval timing summary
    wall_elapsed = time.perf_counter() - eval_t0
    _print_eval_timing_summary(tag, timing_totals, processed, wall_elapsed)
    print(f"\nResults streamed to {csv_path}")

    state_acc = state_correct / state_total if state_total else 0.0
    module_on_acc = (
        module_on_correct / module_on_total if module_on_total else 0.0
    )
    attack_state_success_rate = (
        n_attack_success / n_attack if n_attack else 0.0
    )

    # prints the eval summary
    print(f"\n--- {tag} Summary ---")
    print(f"state accuracy:  {state_correct}/{state_total} = {state_acc * 100:.2f}%")
    if skip_module_analysis:
        print("module-on accuracy:  skipped")
    else:
        print(
            f"module-on accuracy:  {module_on_correct}/{module_on_total} "
            f"= {module_on_acc * 100:.2f}%"
        )
    print(
        f"attack state success rate: "
        f"{n_attack_success}/{n_attack} = {attack_state_success_rate * 100:.2f}%"
        if n_attack else "attack state success rate: n/a"
    )

    state_metrics = compute_state_metrics(metric_rows)
    benign_state_metrics = compute_state_metrics(metric_rows, "benign")
    attack_state_metrics = compute_state_metrics(metric_rows, "attack")

    print_state_metrics_block(f"{tag} State classification (all)", state_metrics)
    print_state_metrics_block(f"{tag} State classification — benign", benign_state_metrics)
    print_state_metrics_block(f"{tag} State classification — attack", attack_state_metrics)

    attack_ranking = _finalize_combo_stats(combo_stats, with_pattern=False)
    pattern_attack_ranking = _finalize_combo_stats(pattern_stats, with_pattern=True)
    print_attack_ranking(tag, attack_ranking, pattern_attack_ranking)

    print("\nPer-variant breakdown:")
    for key in sorted(by_variant):
        v = by_variant[key]
        if key.startswith("benign"):
            print(f"  {key}: state={v['state_correct']}/{v['total']}")
        else:
            print(
                f"  {key}: attack_success={v['attack_success']}/{v['total']} "
                f"state={v['state_correct']}/{v['total']}"
            )

    # Drop metric_rows before returning (caller only needs summary).
    del metric_rows, combo_stats, pattern_stats
    gc.collect()

    summary = {
        "model": os.path.basename(model_path),
        "model_path": model_path,
        "tag": tag,
        "total": total,
        "state_metrics": state_metrics,
        "benign_state_metrics": benign_state_metrics,
        "attack_state_metrics": attack_state_metrics,
        "state_accuracy": state_metrics["accuracy"],
        "state_correct": state_correct, "state_total": state_total,
        "module_on_accuracy": None if skip_module_analysis else module_on_acc,
        "module_on_correct": module_on_correct,
        "module_on_total": module_on_total,
        "module_analysis_skipped": skip_module_analysis,
        "benign_state_accuracy": benign_state_metrics["accuracy"],
        "attack_state_accuracy": attack_state_metrics["accuracy"],
        "attack_state_success_rate": attack_state_success_rate,
        "by_variant": dict(by_variant),
        "attack_ranking": attack_ranking,
        "pattern_attack_ranking": pattern_attack_ranking,
        "most_effective_attack": attack_ranking[0] if attack_ranking else None,
        "least_effective_attack": attack_ranking[-1] if attack_ranking else None,
        "csv_path": csv_path,
        "timing": {
            "model_load_s": model_load_s,
            "eval_wall_s": wall_elapsed,
            "per_stage_s": dict(timing_totals),
            "images_per_s": (processed / wall_elapsed) if wall_elapsed > 0 else 0.0,
            "batch_size": batch_size,
            "prefetch": prefetch,
        },
    }
    if keep_rows:
        summary["rows"] = kept_rows
    return summary

# parses arguments

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate laser injection attacks for benign × laser color combos "
            "and evaluate which combinations fool VLMs on state classification."
        )
    )
    p.add_argument(
        "--benign_source", type=str, choices=BENIGN_SOURCES, default="sim",
        help="Benign image source: sim (pre-cropped) or dtld (demosaiced bbox crops).",
    )
    p.add_argument(
        "--green_benign_dir", type=str,
        default=os.path.join(BASE_DIR, "sim_dataset", "day_go_benign"),
        help="Benign images with green light active (sim source only).",
    )
    p.add_argument(
        "--red_benign_dir", type=str,
        default=os.path.join(BASE_DIR, "sim_dataset", "day_stop_benign"),
        help="Benign images with red light active (sim source only).",
    )
    p.add_argument(
        "--dtld_root", type=str, default=None,
        help="DTLD dataset root (required when --benign_source dtld).",
    )
    p.add_argument(
        "--dtld_split", type=str, choices=DTLD_SPLITS, default="test",
        help="DTLD label split to use when --benign_source dtld.",
    )
    p.add_argument(
        "--dtld_labels", type=str, default=None,
        help="Explicit path to DTLD label JSON/YAML (optional).",
    )
    p.add_argument(
        "--dtld_max_per_class", type=int, default=None,
        help="Max DTLD benign crops per class (green/red); default all.",
    )
    p.add_argument(
        "--max_per_class", type=int, default=None,
        help="Max benign images per class (green/red) for quick runs; applies to sim and DTLD.",
    )
    p.add_argument(
        "--dtld_pad", type=int, default=4,
        help="Pixel padding around DTLD bbox crops.",
    )
    p.add_argument(
        "--attr_filter_json", type=str, default=None,
        help=(
            "JSON object of required DTLD label attributes when --benign_source dtld. "
            f"Default: {DEFAULT_ATTR_FILTERS}"
        ),
    )
    p.add_argument(
        "--no_attr_filter", action="store_true",
        help="Disable DTLD attribute filtering (use all labels matching state split).",
    )
    p.add_argument(
        "--include_unknown", action="store_true",
        help="Include unknown-state DTLD labels (default: exclude).",
    )
    p.add_argument(
        "--realistic_only", action="store_true",
        help=(
            "Only legacy 2-combo realistic attacks: "
            "red laser on green top, green laser on red bottom (no IR). "
            "Overrides the default curated eval combo set."
        ),
    )
    p.add_argument(
        "--all_attack_combos",
        action="store_true",
        help=(
            "Use the full off-module × {red,green,ir} attack matrix (30 combos). "
            f"Default eval/synth uses the curated set of "
            f"{len(DEFAULT_EVAL_ATTACK_COMBOS)} combos from "
            "visualize_laser_injection.ipynb EXAMPLE_ATTACK_CASES."
        ),
    )
    p.add_argument(
        "--attack_combos",
        type=str,
        default=None,
        help=(
            "Override attack combo allowlist: comma-separated combo_keys, or a JSON "
            "list file. Applies to eval (and synth when not --all_attack_combos)."
        ),
    )
    p.add_argument(
        "--max_per_attack_combo",
        type=int,
        default=None,
        help=(
            "Optional cap on attack images evaluated per combo_key after the combo "
            "filter (seeded sample). E.g. 1000 with the default 17-combo filter → "
            "≤17000 attack images. Benign rows are not capped by this flag."
        ),
    )
    p.add_argument(
        "--eval_sample_seed",
        type=int,
        default=42,
        help="RNG seed for --max_per_attack_combo sampling (default: 42).",
    )
    p.add_argument(
        "--benign_only", action="store_true",
        help="Evaluate benign crops only; skip attack synthesis.",
    )
    p.add_argument(
        "--red_laser_dir", type=str,
        default=os.path.join(DEFAULT_LASER_TRACES, "Red"),
        help="Red laser module trace images.",
    )
    p.add_argument(
        "--green_laser_dir", type=str,
        default=os.path.join(DEFAULT_LASER_TRACES, "Green"),
        help="Green laser module trace images.",
    )
    p.add_argument(
        "--ir_laser_dir", type=str,
        default=os.path.join(DEFAULT_LASER_TRACES, "IR_modulated"),
        help="IR laser intensity traces (mag_1..mag_7 in IR_modulated).",
    )
    p.add_argument(
        "--module_analysis_method",
        type=str,
        choices=MODULE_ANALYSIS_METHODS,
        default=DEFAULT_MODULE_ANALYSIS_METHOD,
        help=(
            "Module ON detection method: equal_thirds (per-band CIE-x) or "
            "frame_blob (scene-relative blob analysis; default)."
        ),
    )
    p.add_argument(
        "--skip_module_analysis",
        action="store_true",
        help=(
            "Skip module ON detection (saves time; module-on metrics omitted). "
            "Does not affect state prediction."
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Images per VLM forward pass during evaluation (default: 32).",
    )
    p.add_argument(
        "--progress_every",
        type=int,
        default=5000,
        help="Print progress every N images (default: 5000; was 25).",
    )
    p.add_argument(
        "--prefetch_workers",
        type=int,
        default=2,
        help="Background threads for next-batch crop load (default: 2).",
    )
    p.add_argument(
        "--no_prefetch",
        action="store_true",
        help=(
            "Disable background prefetch of the next batch's crops during inference."
        ),
    )
    p.add_argument(
        "--write_logits",
        action="store_true",
        help=(
            "Include state_probs/logits JSON columns in results CSVs. "
            "Default omits them (much smaller files / less RAM pressure)."
        ),
    )
    p.add_argument(
        "--skip_vlm_samples",
        action="store_true",
        help=(
            "Skip building vlm_samples/ after eval (recommended for large DTLD runs). "
            "Use --extract_vlm_samples_only later if needed."
        ),
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated subset to evaluate: baseline,ontology,ontology_2. "
            "Example for resume after baseline finished: --models ontology,ontology_2"
        ),
    )
    p.add_argument("--output_dir", type=str,
                   default=os.path.join(BASE_DIR, "sim_eval_output"),
                   help="Where to write sample images and results.")
    p.add_argument("--baseline_model", type=str,
                   default=os.path.join(BASE_DIR, "best_model_f1_0.9226.pt"),
                   help="Path to baseline (non-ontology) VLM checkpoint.")
    p.add_argument("--ontology_model", type=str,
                   default=os.path.join(BASE_DIR, "best_model_f1_0.9249.pt"),
                   help="Path to ontology-trained VLM checkpoint.")
    p.add_argument(
        "--ontology_model_2", type=str, default=None,
        help=(
            "Optional second ontology-trained VLM checkpoint. "
            "When set, evaluated as ontology_2 (results_ontology_2.csv)."
        ),
    )
    p.add_argument("--max_save_benign", type=int, default=10,
                   help="Number of benign sample images to save to disk.")
    p.add_argument("--max_save_attack", type=int, default=10,
                   help="Number of attack sample images to save to disk.")
    p.add_argument("--vlm_samples_per_case", type=int, default=3,
                   help="Images saved per most/least effective pattern in each attack case.")
    p.add_argument("--skip_simulation", action="store_true",
                   help="Skip image generation; load existing manifest.json from output_dir.")
    p.add_argument(
        "--synth_cache_dir", type=str, default=None,
        help=(
            "Directory for reusable synthesized DTLD/sim attacks. "
            "If it already contains a valid cache, load it instead of re-synthesizing. "
            "On first run (or with --force_resynth), write generated/ + manifest.json here. "
            "Eval outputs still go to --output_dir."
        ),
    )
    p.add_argument(
        "--force_resynth", action="store_true",
        help="Ignore existing --synth_cache_dir contents and rebuild the attack cache.",
    )
    p.add_argument(
        "--synth_only", action="store_true",
        help=(
            "Only synthesize/cache attacks (requires --synth_cache_dir); skip VLM evaluation. "
            "Useful for a one-time DTLD synthesis job."
        ),
    )
    p.add_argument("--extract_vlm_samples_only", action="store_true",
                   help="Skip inference; build vlm_samples/ from existing CSV results.")
    return p.parse_args()


def print_benign_comparison(summaries):
    """Print benign-only metrics side-by-side for attack comparison."""
    print(f"\n{'=' * 60}")
    print("BENIGN BASELINE (no attack synthesis)")
    print(f"{'=' * 60}")

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
        ("benign state accuracy", "benign_state_metrics", "{:.2%}", "accuracy"),
        ("benign state macro f1", "benign_state_metrics", "{:.3f}", "macro_f1"),
    ]
    for label, key, fmt, subkey in metrics:
        cells = [
            fmt.format(summaries.get(tag, {}).get(key, {}).get(subkey, 0))
            for tag in tags
        ]
        print(f"{label:<28}" + "".join(f"{c:>{col_w}}" for c in cells))

# main

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.extract_vlm_samples_only:
        print(f"Extracting VLM samples from existing results in {args.output_dir}")
        pointer = load_manifest_meta(args.output_dir)
        if pointer.get("synth_cache_dir"):
            manifest = list(iter_manifest_entries(pointer["synth_cache_dir"]))
        elif resolve_manifest_jsonl(args.output_dir):
            manifest = list(iter_manifest_entries(args.output_dir))
        else:
            manifest = load_manifest(args.output_dir)
        extract_vlm_samples_from_csv(
            args.output_dir, manifest, samples_per_pattern=args.vlm_samples_per_case
        )
        return

    if not args.synth_only:
        model_checks = [
            ("baseline", args.baseline_model),
            ("ontology", args.ontology_model),
        ]
        if args.ontology_model_2:
            model_checks.append(("ontology_2", args.ontology_model_2))
        for label, path in model_checks:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{label} model checkpoint not found: {path}\n"
                    "Train models with 4_1-train_vlm.py (baseline) and "
                    "4_2-train_vlm_onto.py (ontology) first."
                )

    manifest_root = None
    n_entries = None
    manifest_list = None

    if args.skip_simulation:
        print(f"Resolving existing manifest from {args.output_dir} (streamed)")
        # Defer actual streaming to Step 2; do not load entries here.
    else:
        mode_bits = []
        if args.benign_source == "dtld":
            mode_bits.append("DTLD benign")
        else:
            mode_bits.append("sim benign")
        if args.benign_only:
            mode_bits.append("benign-only eval")
        elif args.realistic_only:
            mode_bits.append("realistic attacks (2 combos)")
        elif args.all_attack_combos:
            mode_bits.append("full off-module attack matrix")
        else:
            mode_bits.append(
                f"eval-default attacks ({len(DEFAULT_EVAL_ATTACK_COMBOS)} combos)"
            )
        print(f"=== Step 1: Preparing dataset ({', '.join(mode_bits)}) ===")

        if args.benign_source == "dtld":
            if not args.dtld_root:
                raise ValueError("--dtld_root is required when --benign_source dtld")
            if not os.path.isdir(args.dtld_root):
                raise FileNotFoundError(f"dtld_root not found: {args.dtld_root}")
        else:
            for label, path in [
                ("green_benign_dir", args.green_benign_dir),
                ("red_benign_dir", args.red_benign_dir),
            ]:
                if not os.path.isdir(path):
                    raise FileNotFoundError(f"{label} not found: {path}")

        if not args.benign_only:
            laser_checks = [("red_laser_dir", args.red_laser_dir), ("green_laser_dir", args.green_laser_dir)]
            if not args.realistic_only:
                laser_checks.append(("ir_laser_dir", args.ir_laser_dir))
            for label, path in laser_checks:
                if not os.path.isdir(path):
                    raise FileNotFoundError(f"{label} not found: {path}")

        if args.synth_only and not args.synth_cache_dir:
            raise ValueError("--synth_only requires --synth_cache_dir")

        fingerprint = build_synth_cache_fingerprint(args)
        cache_dir = _abspath(args.synth_cache_dir) if args.synth_cache_dir else None


        if (
            cache_dir
            and synth_cache_ready(cache_dir)
            and not args.force_resynth
        ):
            print(f"Reusing synth cache: {cache_dir}")
            if args.synth_only:
                print("synth_only: cache already present; nothing to do")
                return
            load_synth_cache(cache_dir, expected_fingerprint=fingerprint)
            n_entries = count_manifest_entries(cache_dir)
            write_eval_manifest_pointer(args.output_dir, cache_dir, n_entries, args)
            manifest_root = cache_dir
        else:
            synth_out = cache_dir if cache_dir else args.output_dir
            if cache_dir:
                if args.force_resynth and os.path.isdir(cache_dir):
                    print(f"Force-resynth: clearing {cache_dir}")
                    shutil.rmtree(cache_dir)
                print(f"Synthesizing attacks into cache: {synth_out}")
            else:
                print(f"Synthesizing attacks into output_dir: {synth_out}")

            generate_simulated_dataset(
                green_benign_dir=args.green_benign_dir,
                red_benign_dir=args.red_benign_dir,
                red_laser_dir=args.red_laser_dir,
                green_laser_dir=args.green_laser_dir,
                ir_laser_dir=args.ir_laser_dir,
                output_dir=synth_out,
                max_save_benign=args.max_save_benign,
                max_save_attack=args.max_save_attack,
                benign_source=args.benign_source,
                dtld_root=args.dtld_root,
                dtld_split=args.dtld_split,
                dtld_labels=args.dtld_labels,
                dtld_max_per_class=args.dtld_max_per_class,
                max_per_class=args.max_per_class,
                dtld_pad=args.dtld_pad,
                dtld_attr_filters={} if args.no_attr_filter else load_attr_filters(args.attr_filter_json),
                dtld_exclude_unknown=not args.include_unknown,
                realistic_only=args.realistic_only,
                eval_default=(
                    not args.realistic_only and not args.all_attack_combos
                ),
                benign_only=args.benign_only,
                return_entries=False,
            )
            if cache_dir:
                finalize_synth_cache(cache_dir, fingerprint, return_entries=False)
                n_entries = count_manifest_entries(cache_dir)
                if cache_dir != _abspath(args.output_dir):
                    write_eval_manifest_pointer(args.output_dir, cache_dir, n_entries, args)
                manifest_root = cache_dir
            else:
                n_entries = count_manifest_entries(args.output_dir)
                manifest_root = args.output_dir

    if args.synth_only:
        print("synth_only: skipping VLM evaluation")
        return

    # Resolve where to stream entries from (never hold full manifest in RAM).
    if args.skip_simulation or manifest_root is None:
        # Prefer explicit --synth_cache_dir (avoids parsing multi-GB output manifests).
        if args.synth_cache_dir and synth_cache_ready(args.synth_cache_dir):
            manifest_root = _abspath(args.synth_cache_dir)
            n_entries = count_manifest_entries(manifest_root)
            write_eval_manifest_pointer(args.output_dir, manifest_root, n_entries, args)
            print(f"Streaming attacks from --synth_cache_dir: {manifest_root}")
        elif args.skip_simulation:
            try:
                pointer = load_manifest_meta(args.output_dir)
            except ValueError as exc:
                raise ValueError(
                    f"{exc}\nHint: re-run with --synth_cache_dir "
                ) from exc
            if pointer.get("synth_cache_dir"):
                manifest_root = pointer["synth_cache_dir"]
                n_entries = pointer.get("n_entries") or count_manifest_entries(manifest_root)
            elif resolve_manifest_jsonl(args.output_dir):
                manifest_root = args.output_dir
                n_entries = count_manifest_entries(manifest_root)
            else:
                print(
                    "WARNING: loading monolithic manifest.json into RAM "
                    "(prefer --synth_cache_dir with manifest_entries.jsonl)."
                )
                manifest_list = load_manifest(args.output_dir)
                n_entries = len(manifest_list)
                manifest_root = None
    # else: manifest_root / n_entries already set by synth/cache branches

    attack_allowlist = resolve_attack_combo_allowlist(args)
    if attack_allowlist is None:
        print("Attack combo filter: ALL (--all_attack_combos)")
    else:
        print(
            f"Attack combo filter: {len(attack_allowlist)} curated combo(s) "
            "(default eval set; pass --all_attack_combos for full matrix)"
        )
        for k in sorted(attack_allowlist):
            print(f"  {k}")

    max_per_combo = getattr(args, "max_per_attack_combo", None)
    eval_seed = int(getattr(args, "eval_sample_seed", 42))
    if max_per_combo is not None and max_per_combo > 0:
        n_combos = (
            len(attack_allowlist)
            if attack_allowlist is not None
            else "all"
        )
        print(
            f"Attack sample cap: <={max_per_combo} image(s) per combo "
            f"(seed={eval_seed}; ~{max_per_combo}x{n_combos} attacks max)"
        )

    # recount after combo filter (optional per-combo cap)
    if manifest_root is not None:
        n_entries = count_filtered_manifest_entries(
            manifest_root,
            attack_allowlist,
            max_per_attack_combo=max_per_combo,
            seed=eval_seed,
        )
    elif manifest_list is not None:
        manifest_list = [
            e for e in manifest_list if manifest_entry_allowed(e, attack_allowlist)
        ]
        manifest_list = cap_entries_per_attack_combo(
            manifest_list, max_per_combo, seed=eval_seed
        )
        n_entries = len(manifest_list)

    n_models = 3 if args.ontology_model_2 else 2
    print(f"\n=== Step 2: Evaluating {n_entries} images with {n_models} models ===")
    module_method = None if args.skip_module_analysis else args.module_analysis_method
    print(
        f"Module analysis method: "
        f"{'skipped' if module_method is None else module_method}"
    )
    print(f"Batch size: {args.batch_size}")
    print(f"Manifest stream root: {manifest_root or args.output_dir}")
    if not args.no_prefetch:
        print("Prefetch: on (next batch loads during inference)")
    if not args.write_logits:
        print("CSV: omitting logits/state_probs (pass --write_logits to include)")

    eval_kwargs = {
        "module_analysis_method": module_method,
        "batch_size": args.batch_size,
        "prefetch": not args.no_prefetch,
        "prefetch_workers": int(getattr(args, "prefetch_workers", 2)),
        "progress_every": int(getattr(args, "progress_every", 5000)),
        "total": n_entries,
        "omit_heavy_fields": not args.write_logits,
        "keep_rows": False,
    }

    def _manifest_iter():
        if manifest_root is not None:
            return iter_filtered_manifest_entries(
                manifest_root,
                attack_allowlist,
                check_paths=False,
                max_per_attack_combo=max_per_combo,
                seed=eval_seed,
            )
        return iter(manifest_list)
    summaries = {}
    model_jobs = [
        ("baseline", args.baseline_model, "BASELINE", "results_baseline.csv"),
        ("ontology", args.ontology_model, "ONTOLOGY", "results_ontology.csv"),
    ]
    if args.ontology_model_2:
        model_jobs.append(
            ("ontology_2", args.ontology_model_2, "ONTOLOGY_2", "results_ontology_2.csv")
        )
    if args.models:
        wanted = {m.strip().lower() for m in args.models.split(",") if m.strip()}
        model_jobs = [job for job in model_jobs if job[0] in wanted]
        if not model_jobs:
            raise ValueError(f"--models matched nothing: {args.models!r}")
        print(f"Evaluating subset via --models: {[j[0] for j in model_jobs]}")

    for key, model_path, tag, csv_name in model_jobs:
        summaries[key] = evaluate_manifest(
            _manifest_iter(),
            model_path=model_path,
            tag=tag,
            csv_path=os.path.join(args.output_dir, csv_name),
            **eval_kwargs,
        )
        # Drop any accidental heavy refs before next model.
        summaries[key].pop("rows", None)
        gc.collect()

    # Fill any missing models from prior summary.json and/or existing results_*.csv
    # (needed after OOM: baseline CSV exists but summary was never written).
    csv_by_tag = {
        "baseline": ("results_baseline.csv", "BASELINE", getattr(args, "baseline_model", "")),
        "ontology": ("results_ontology.csv", "ONTOLOGY", getattr(args, "ontology_model", "")),
        "ontology_2": ("results_ontology_2.csv", "ONTOLOGY_2", getattr(args, "ontology_model_2", "") or ""),
    }
    prior = {}
    prior_summary_path = os.path.join(args.output_dir, "summary.json")
    if os.path.isfile(prior_summary_path):
        try:
            with open(prior_summary_path, "r", encoding="utf-8") as f:
                prior = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARNING: could not read prior summary.json: {exc}")

    for key, (csv_name, tag, model_path) in csv_by_tag.items():
        if key in summaries:
            continue
        if isinstance(prior.get(key), dict) and prior[key].get("attack_ranking"):
            summaries[key] = prior[key]
            print(f"Merged prior summary for skipped model: {key}")
            continue
        csv_path = os.path.join(args.output_dir, csv_name)
        if os.path.isfile(csv_path):
            print(f"Rebuilding {key} summary/rankings from {csv_path} ...")
            summaries[key] = rebuild_summary_from_results_csv(
                csv_path, tag=tag, model_path=model_path or ""
            )
        elif key == "ontology_2" and not args.ontology_model_2:
            continue

    print_comparison(summaries)
    print_benign_comparison(summaries)
    if not args.benign_only:
        print_cross_model_attack_ranking(summaries)

    attack_report = {}
    if not args.benign_only:
        attack_report = {
            tag: {
                "attack_ranking": summaries[tag].get("attack_ranking", []),
                "pattern_attack_ranking": summaries[tag].get("pattern_attack_ranking", []),
                "most_effective_attack": summaries[tag].get("most_effective_attack"),
                "least_effective_attack": summaries[tag].get("least_effective_attack"),
                "overall_attack_state_success_rate": summaries[tag].get(
                    "attack_state_success_rate", 0.0
                ),
            }
            for tag in summaries
        }

    summary_path = os.path.join(args.output_dir, "summary.json")
    serializable_summaries = {
        tag: {k: v for k, v in summary.items() if k != "rows"}
        for tag, summary in summaries.items()
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            **serializable_summaries,
            "attack_report": attack_report,
            "benign_baseline": {
                tag: {
                    "benign_state_metrics": summaries[tag].get("benign_state_metrics", {}),
                }
                for tag in summaries
            },
            "config": {
                "benign_source": args.benign_source,
                "realistic_only": args.realistic_only,
                "all_attack_combos": bool(args.all_attack_combos),
                "attack_combo_allowlist": (
                    None if attack_allowlist is None else sorted(attack_allowlist)
                ),
                "max_per_attack_combo": max_per_combo,
                "eval_sample_seed": eval_seed,
                "benign_only": args.benign_only,
                "ontology_model_2": args.ontology_model_2,
                "skip_module_analysis": args.skip_module_analysis,
                "module_analysis_method": module_method,
                "batch_size": args.batch_size,
                "prefetch": not args.no_prefetch,
                "omit_heavy_csv_fields": not args.write_logits,
                "skip_vlm_samples": args.skip_vlm_samples,
            },
        }, f, indent=2)
    print(f"\nCombined summary saved to {summary_path}")

    if attack_report:
        attack_report_path = os.path.join(args.output_dir, "attack_ranking.json")
        with open(attack_report_path, "w", encoding="utf-8") as f:
            json.dump(attack_report, f, indent=2)
        print(f"Attack ranking saved to {attack_report_path}")

    if args.skip_vlm_samples or n_entries >= 100_000:
        reason = "--skip_vlm_samples" if args.skip_vlm_samples else f"n_entries={n_entries} >= 100000"
        print(
            f"\n=== Step 3: Skipping VLM sample pack ({reason}) ===\n"
            "Re-run with --extract_vlm_samples_only after eval if needed."
        )
    else:
        print("\n=== Step 3: Saving VLM output samples (from CSVs) ===")
        # Small runs only — rebuild from CSV without keeping eval rows in RAM.
        if manifest_root is not None:
            manifest_for_samples = list(
                iter_filtered_manifest_entries(
                    manifest_root,
                    attack_allowlist,
                    max_per_attack_combo=max_per_combo,
                    seed=eval_seed,
                )
            )
        else:
            manifest_for_samples = manifest_list
        extract_vlm_samples_from_csv(
            args.output_dir,
            manifest_for_samples,
            samples_per_pattern=args.vlm_samples_per_case,
        )



if __name__ == "__main__":
    main()
