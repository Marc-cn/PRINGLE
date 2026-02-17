# make_json_from_crops.py
import os, re, json, argparse
from pathlib import Path
from PIL import Image

CANON = {"red","yellow","green","red_yellow","off"}

def infer_state(p: Path):
    # 1) parent folder name
    s = p.parent.name.lower()
    if s in CANON:
        return s
    # 2) filename pattern: ...__{state}__...
    m = re.search(r"__(red_yellow|red-yellow|red_amber|red|yellow|green|off)__", p.name.lower())
    if m:
        s = m.group(1).replace("red-yellow","red_yellow").replace("red_amber","red_yellow")
        return s if s in CANON else None
    return None

def infer_tag(p: Path):
    # from crop filename: <tag>__img00001__box...
    m = re.match(r"([^_\\/:]+)__img", p.name)
    return m.group(1) if m else "crops"

def main():
    ap = argparse.ArgumentParser("Build DTLD-style JSONs for pre-cropped JPGs")
    ap.add_argument("--crops_root", required=True, help="Root folder with saved crops (e.g., ./crops_jpg)")
    ap.add_argument("--out_dir", required=True, help="Where to write JSONs")
    args = ap.parse_args()

    crops_root = Path(args.crops_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = {}  # tag -> list of image entries
    for p in crops_root.rglob("*.jpg"):
        state = infer_state(p)
        if state is None:
            print(f"[WARN] Could not infer state for {p}")
            continue
        try:
            with Image.open(p) as im:
                W, H = im.size
        except Exception as e:
            print(f"[WARN] Failed to open {p}: {e}")
            continue

        rel = p.relative_to(crops_root).as_posix()
        entry = {
            "image_path": rel,
            "labels": [{
                "x": 0, "y": 0, "w": W, "h": H,
                "attributes": {"state": state}
            }]
        }
        tag = infer_tag(p)
        buckets.setdefault(tag, []).append(entry)

    # write one JSON per tag
    written = []
    for tag, items in buckets.items():
        js = {"images": items}
        out_path = out_dir / f"{tag}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(js, f, ensure_ascii=False)
        written.append(out_path)
        print(f"[OK] {tag}: {len(items)} crops -> {out_path}")

    print(f"[DONE] Wrote {len(written)} JSONs to {out_dir}")

if __name__ == "__main__":
    main()
