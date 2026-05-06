# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pathlib import Path

import seq1 import find_paper_mask
import seq1 import find_dot
from seq1 import COLOR_RANGES

from seq3 import find_hand_mask


# ── Global correction store ───────────────────────────────────────────────────
_correction_stats = {
    "yellow": [], "red": [], "blue": [], "green": []
}

def parallelogram_predict(color, tl, tr, bl, br):
    if color == "yellow":
        return (tr[0] + bl[0] - br[0], tr[1] + bl[1] - br[1])
    elif color == "red":
        return (tl[0] + br[0] - bl[0], tl[1] + br[1] - bl[1])
    elif color == "blue":
        return (tl[0] + br[0] - tr[0], tl[1] + br[1] - tr[1])
    else:  # green
        return (tr[0] + bl[0] - tl[0], tr[1] + bl[1] - tl[1])


def update_correction_stats(tl, tr, bl, br):
    positions = {"yellow": tl, "red": tr, "blue": bl, "green": br}
    for color, true_pos in positions.items():
        pred = parallelogram_predict(color, tl, tr, bl, br)
        err = (true_pos[0] - pred[0], true_pos[1] - pred[1])
        _correction_stats[color].append(err)


def get_correction(color):
    errors = _correction_stats[color]
    if not errors:
        return (0, 0)
    return errors[-1]  # last observed error only



def find_dots_with_occlusion(hsv_img, paper_mask):
    dot_colors = ["yellow", "red", "blue", "green"]
    dot_positions = {}

    for color in dot_colors:
        try:
            dot_positions[color] = find_dot(hsv_img, color, paper_mask)
        except RuntimeError:
            print(f"  [{color}] not found — will infer.")
            dot_positions[color] = None

    missing = [c for c, v in dot_positions.items() if v is None]

    if len(missing) == 0:
        update_correction_stats(
            dot_positions["yellow"], dot_positions["red"],
            dot_positions["blue"],   dot_positions["green"]
        )

    elif len(missing) == 1:
        color = missing[0]
        tl = dot_positions["yellow"]
        tr = dot_positions["red"]
        bl = dot_positions["blue"]
        br = dot_positions["green"]

        pred = parallelogram_predict(color, tl, tr, bl, br)
        correction = get_correction(color)
        inferred = (
            int(round(pred[0] + correction[0])),
            int(round(pred[1] + correction[1]))
        )
        print(f"  [{color}] pred={pred}, correction={correction}, final={inferred}")
        dot_positions[color] = inferred

    else:
        raise RuntimeError(f"Too many dots occluded ({missing}).")

    return (
        dot_positions["yellow"],
        dot_positions["red"],
        dot_positions["blue"],
        dot_positions["green"],
    )



# ── Camera calibration parameters ────────────────────────────────────────────
K = np.array([
    [533.75781056,   0.          , 386.78762246],
    [  0.          , 534.74578856, 275.71106165],
    [  0.          ,   0.        ,   1.        ]
], dtype=np.float64)

DIST = np.array(
    [-3.33535276e-01,  1.65338810e-01, -2.90030682e-04, -3.97059918e-04, -4.70631813e-02],
    dtype=np.float64
)

def undistort(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    # Compute the optimal new camera matrix.
    # alpha=0 (no black borders)
    # alpha=1 keep all pixels
    alpha = 0
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, DIST, (w, h), alpha)

    # Undistort
    dst = cv2.undistort(img, K, DIST, None, new_K)

    # Crop to the valid ROI returned by getOptimalNewCameraMatrix
    # Only crop if alpha < 1
    if alpha < 1.0:
        x, y, rw, rh = roi
        if rw > 0 and rh > 0:
            dst = dst[y:y + rh, x:x + rw]

    return dst



def overlay_image_with_correction(background_path, overlay_path, output_path, correction_fn):
    bg  = cv2.imread(background_path)
    ovr = cv2.imread(overlay_path)
    bg  = undistort(bg)

    hsv        = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    paper_mask = find_paper_mask(bg)

    dot_positions = {}
    missing = []
    for color in ["yellow", "red", "blue", "green"]:
        try:
            dot_positions[color] = find_dot(hsv, color, paper_mask)
        except RuntimeError:
            dot_positions[color] = None
            missing.append(color)

    if len(missing) > 1:
        raise RuntimeError(f"Too many dots occluded: {missing}")

    if len(missing) == 1:
        color = missing[0]
        tl = dot_positions["yellow"]
        tr = dot_positions["red"]
        bl = dot_positions["blue"]
        br = dot_positions["green"]

        pred       = parallelogram_predict(color, tl, tr, bl, br)
        correction = correction_fn(color)
        dot_positions[color] = (
            int(round(pred[0] + correction[0])),
            int(round(pred[1] + correction[1])),
        )
        print(f"  [{color}] interpolated correction={correction}")

    tl = dot_positions["yellow"]
    tr = dot_positions["red"]
    bl = dot_positions["blue"]
    br = dot_positions["green"]

    dst_pts = np.array([tl, tr, br, bl], dtype=np.float32)
    oh, ow  = ovr.shape[:2]
    src_pts = np.array([[0,0],[ow-1,0],[ow-1,oh-1],[0,oh-1]], dtype=np.float32)

    H      = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(ovr, H, (bg.shape[1], bg.shape[0]))

    quad_mask = np.zeros((bg.shape[0], bg.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(quad_mask, dst_pts.astype(int), 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    quad_mask_enlarged = cv2.dilate(quad_mask, kernel, iterations=1)

    hand_mask = find_hand_mask(bg, quad_mask_enlarged)

    quad_3ch = cv2.merge([quad_mask]*3)
    hand_3ch = cv2.merge([hand_mask]*3)

    result = np.where(quad_3ch == 255, warped, bg)
    result = np.where(hand_3ch == 255, bg, result)

    cv2.imwrite(output_path, result)
    print(f"Saved → {output_path}")
    


# ── Batch processing ──────────────────────────────────────────────────────────
def process_batch(overlay_path: str, backgrounds: list, output_dir: str = "output"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ovr = cv2.imread(overlay_path)
    if ovr is None:
        raise FileNotFoundError(f"Overlay not found: {overlay_path}")

    # ── Pass 1: collect per-frame corrections where all 4 dots are visible
    print("Pass 1: collecting corrections...")
    frame_corrections = {}

    for i, bg_path in enumerate(backgrounds):
        bg = cv2.imread(bg_path)
        if bg is None:
            continue
        bg = undistort(bg)
        hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        paper_mask = find_paper_mask(bg)

        dot_positions = {}
        all_found = True
        for color in ["yellow", "red", "blue", "green"]:
            try:
                dot_positions[color] = find_dot(hsv, color, paper_mask)
            except RuntimeError:
                all_found = False
                break

        if all_found:
            tl = dot_positions["yellow"]
            tr = dot_positions["red"]
            bl = dot_positions["blue"]
            br = dot_positions["green"]
            frame_corrections[i] = {
                color: (
                    dot_positions[color][0] - parallelogram_predict(color, tl, tr, bl, br)[0],
                    dot_positions[color][1] - parallelogram_predict(color, tl, tr, bl, br)[1],
                )
                for color in ["yellow", "red", "blue", "green"]
            }
            print(f"  frame {i:04d}: all dots found, corrections recorded")
        else:
            frame_corrections[i] = None
            print(f"  frame {i:04d}: occlusion detected")

    # ── Interpolate corrections for occluded frames ──────────────────────────
    known_indices = sorted(k for k, v in frame_corrections.items() if v is not None)

    def interpolate_correction(frame_idx, color):
        """Linear interpolation between nearest known corrections before and after."""
        before = [k for k in known_indices if k <= frame_idx]
        after  = [k for k in known_indices if k >= frame_idx]

        if before and after:
            k0, k1 = before[-1], after[0]
            if k0 == k1:
                return frame_corrections[k0][color]
            # linear interpolation
            t = (frame_idx - k0) / (k1 - k0)
            c0 = frame_corrections[k0][color]
            c1 = frame_corrections[k1][color]
            return (c0[0] + t * (c1[0] - c0[0]),
                    c0[1] + t * (c1[1] - c0[1]))
        elif before:
            return frame_corrections[before[-1]][color]  # extrapolate flat
        elif after:
            return frame_corrections[after[0]][color]    # extrapolate flat
        else:
            return (0, 0)

    # ── Pass 2: render with interpolated corrections ─────────────────────────
    print("\nPass 2: rendering...")
    for i, bg_path in enumerate(backgrounds):
        stem = Path(bg_path).stem
        out  = str(Path(output_dir) / f"{stem}_result.jpg")
        try:
            overlay_image_with_correction(
                bg_path, overlay_path, out,
                correction_fn=lambda color, i=i: interpolate_correction(i, color)
            )
        except Exception as e:
            print(f"[ERROR] {bg_path}: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import glob

    OVERLAY     = "koala.jpg"
    BACKGROUNDS = sorted(glob.glob("seq4/*.png"))
    OUTPUT_DIR  = "output"

    process_batch(OVERLAY, BACKGROUNDS, OUTPUT_DIR)
