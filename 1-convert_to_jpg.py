# tiff_to_jpg_recursive.py
import os, glob
import numpy as np
import cv2

# -----------------------------
# EDIT THIS: set your dataset roots
# -----------------------------
ROOTS = [
    r"PATH_TO_YOUR_DATASET_ROOT_1",  # e.g., r"C:\data\DTLD\Bochum1" or r"/data/dtld/Bochum1"
    # r"PATH_TO_YOUR_DATASET_ROOT_2",
    # r"PATH_TO_YOUR_DATASET_ROOT_3",
]

def stretch_16bit_to_8bit(img16, low_p=2, high_p=98, gamma=1.0):
    """Percentile stretch 16-bit/float to 0..255 uint8, optional gamma."""
    img = img16.astype(np.float32)
    p_lo, p_hi = np.percentile(img, (low_p, high_p))
    if p_hi <= p_lo:  # degenerate, fallback to min/max
        p_lo, p_hi = img.min(), img.max()
    img = np.clip((img - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    if gamma != 1.0:
        img = np.power(img, gamma)
    return (img * 255.0 + 0.5).astype(np.uint8)

def enhance_gray(gray8):
    """CLAHE to improve contrast on 8-bit grayscale."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray8)

def save_jpg(path, img_bgr_or_gray):
    ok = cv2.imwrite(path, img_bgr_or_gray, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        print(f"[ERROR] Failed to save {path}")

def process_folder(FOLDER):
    # find tiffs in this folder (non-recursive)
    tifs = glob.glob(os.path.join(FOLDER, "*.tif")) + glob.glob(os.path.join(FOLDER, "*.tiff"))
    if not tifs:
        return  # nothing to do here

    print(f"[INFO] Processing folder: {FOLDER}")

    # 0) remove previous JPGs (only next to the TIFFs)
    for jpg in glob.glob(os.path.join(FOLDER, "*.jpg")):
        try:
            os.remove(jpg)
        except Exception as e:
            print(f"[WARN] Could not delete {jpg}: {e}")

    # 1) convert all TIFFs in the folder (non-recursive)
    for tif in tifs:
        base = os.path.splitext(os.path.basename(tif))[0]
        out = os.path.join(FOLDER, base + ".jpg")

        # read preserving depth
        img = cv2.imread(tif, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Could not read {tif}")
            continue

        name_lower = base.lower()

        # A) Left/right intensity frames: *_k0.tiff / *_k1.tiff
        if name_lower.endswith("_k0") or name_lower.endswith("_k1"):
            if img.dtype == np.uint8:
                gray8 = img
            else:
                gray8 = stretch_16bit_to_8bit(img, low_p=2, high_p=98, gamma=0.9)
            gray8 = enhance_gray(gray8)
            save_jpg(out, gray8)

        # B) Disparity frames: *_nativeV2.tiff  (visualize with a heatmap)
        elif name_lower.endswith("_nativev2"):
            disp = img.astype(np.float32)

            invalid_mask = (disp <= 0) | (disp >= np.max(disp))
            valid = disp[~invalid_mask]
            if valid.size == 0:
                print(f"[WARN] No valid disparity in {tif}")
                continue
            d8 = stretch_16bit_to_8bit(valid, low_p=1, high_p=99)
            vis = np.zeros_like(disp, dtype=np.uint8)
            vis[~invalid_mask] = d8
            vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            vis_color[invalid_mask] = (0,0,0)
            save_jpg(out, vis_color)

        # C) anything else: safe visibility pass
        else:
            if img.ndim == 2:
                if img.dtype != np.uint8:
                    img8 = stretch_16bit_to_8bit(img, low_p=2, high_p=98, gamma=0.9)
                else:
                    img8 = img
                img8 = enhance_gray(img8)
                save_jpg(out, img8)
            else:
                if img.dtype != np.uint8:
                    chans = [stretch_16bit_to_8bit(img[...,c], low_p=2, high_p=98, gamma=0.9) for c in range(img.shape[2])]
                    img8 = cv2.merge(chans)
                else:
                    img8 = img
                save_jpg(out, img8)

def main():
    # Safety check: prevent accidentally committing your real dataset paths
    if not ROOTS or any("PATH_TO_YOUR_DATASET_ROOT" in r for r in ROOTS):
        print(
            "[ERROR] Please edit ROOTS in this script and replace PATH_TO_YOUR_DATASET_ROOT_* "
            "with your local dataset directories."
        )
        return

    # Walk all subfolders; convert per-folder where TIFFs are present.
    for root in ROOTS:
        if not os.path.isdir(root):
            print(f"[WARN] Missing root: {root}")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # process only folders that contain TIFFs
            if any(f.lower().endswith((".tif", ".tiff")) for f in filenames):
                process_folder(dirpath)

    print("Done.")

if __name__ == "__main__":
    main()
