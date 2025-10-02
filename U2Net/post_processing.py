import os
import glob
import cv2
import numpy as np

# === Your folders ===
IN_DIR  = r"C:\Users\bob1k\Desktop\MASTER\Thesis\hed_project\saved_models\u2netv4"
OUT_DIR = r"C:\Users\bob1k\Desktop\MASTER\Thesis\hed_project\saved_models\u2netv4_PostProcessed"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Tunables (adjust if needed) ---
MAX_EDGE_THICK = 8    # max thickness (px) of each edge line we accept
MAX_GAP        = 60   # max distance (px) between the two edges to fill
SMOOTH_ITERS   = 1    # small closing to seal 1px gaps in the filled area

# ---------- Edge detection (robust for float inputs + soft edges) ----------
def canny_edges_strong_or_float_safe(img: np.ndarray) -> np.ndarray:
    # Ensure uint8 0..255
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1) if img.max() <= 1.0 else np.clip(img / 255.0, 0, 1)
        img = (img * 255).astype(np.uint8)

    blur = cv2.GaussianBlur(img, (3, 3), 0)

    gx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(
        cv2.addWeighted(cv2.convertScaleAbs(gx), 1.0, cv2.convertScaleAbs(gy), 1.0, 0)
    )
    otsu, _ = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low  = int(max(5, otsu * 0.5))
    high = int(max(low + 1, otsu * 1.5))

    edges = cv2.Canny(blur, low, high)
    return (edges > 0).astype(np.uint8) * 255  # strict binary

# ---------- Scanline fill between two edges ----------
def _runs_1d(binary_line: np.ndarray):
    """Return [(start,end)] for each run of 1s in a 1D binary array."""
    ones = np.where(binary_line > 0)[0]
    if len(ones) == 0:
        return []
    # split into runs
    split_ix = np.where(np.diff(ones) > 1)[0] + 1
    groups = np.split(ones, split_ix)
    runs = [(g[0], g[-1]) for g in groups]
    return runs

def _fill_between_on_axis(edges: np.ndarray, axis: int,
                          max_edge_thick: int, max_gap: int) -> np.ndarray:
    """
    Fill between two edge runs along rows (axis=0) or columns (axis=1).
    Returns a binary mask of the filled area.
    """
    h, w = edges.shape
    filled = np.zeros_like(edges, dtype=np.uint8)

    if axis == 0:  # row-wise
        for y in range(h):
            runs = _runs_1d(edges[y, :])
            if len(runs) < 2:
                continue
            # Keep only thin runs (edge-like)
            runs = [r for r in runs if (r[1] - r[0] + 1) <= max_edge_thick]
            if len(runs) < 2:
                continue
            # Consider adjacent run pairs and pick the closest ones
            best_pair = None
            best_gap = None
            for (a0, a1), (b0, b1) in zip(runs[:-1], runs[1:]):
                gap = max(0, b0 - a1 - 1)
                if gap <= max_gap:
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_pair = (a1 + 1, b0 - 1)
            if best_pair and best_pair[0] <= best_pair[1]:
                filled[y, best_pair[0]:best_pair[1] + 1] = 255
    else:  # column-wise
        for x in range(w):
            runs = _runs_1d(edges[:, x])
            if len(runs) < 2:
                continue
            runs = [r for r in runs if (r[1] - r[0] + 1) <= max_edge_thick]
            if len(runs) < 2:
                continue
            best_pair = None
            best_gap = None
            for (a0, a1), (b0, b1) in zip(runs[:-1], runs[1:]):
                gap = max(0, b0 - a1 - 1)
                if gap <= max_gap:
                    if best_gap is None or gap < best_gap:
                        best_gap = gap
                        best_pair = (a1 + 1, b0 - 1)
            if best_pair and best_pair[0] <= best_pair[1]:
                filled[best_pair[0]:best_pair[1] + 1, x] = 255

    return filled

def fill_between_two_edges(edges: np.ndarray,
                           max_edge_thick: int = MAX_EDGE_THICK,
                           max_gap: int = MAX_GAP,
                           smooth_iters: int = SMOOTH_ITERS) -> np.ndarray:
    """
    Fill the region between two nearby edge lines using row- and column-wise scan.
    Returns strictly binary (0/255).
    """
    row_fill = _fill_between_on_axis(edges, axis=0,
                                     max_edge_thick=max_edge_thick, max_gap=max_gap)
    col_fill = _fill_between_on_axis(edges, axis=1,
                                     max_edge_thick=max_edge_thick, max_gap=max_gap)
    filled = cv2.bitwise_or(row_fill, col_fill)

    # Small morphological closing to seal 1px holes; keep binary output
    if smooth_iters > 0:
        k = np.ones((3, 3), np.uint8)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k, iterations=smooth_iters)

    return (filled > 0).astype(np.uint8) * 255

# ---------- I/O ----------
def process_one(in_path: str, out_dir: str):
    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] Could not read: {in_path}")
        return

    edges = canny_edges_strong_or_float_safe(img)
    filled = fill_between_two_edges(edges,
                                    max_edge_thick=MAX_EDGE_THICK,
                                    max_gap=MAX_GAP,
                                    smooth_iters=SMOOTH_ITERS)

    base, ext = os.path.splitext(os.path.basename(in_path))
    cv2.imwrite(os.path.join(out_dir, f"{base}_edges{ext}"), edges)
    cv2.imwrite(os.path.join(out_dir, f"{base}_filled{ext}"), filled)

def main():
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(IN_DIR, p)))
    if not files:
        print(f"[INFO] No images found in {IN_DIR}")
        return

    print(f"[INFO] Processing {len(files)} files...")
    for fp in files:
        process_one(fp, OUT_DIR)
        print(f"[OK] {os.path.basename(fp)}")
    print("[DONE] Saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
