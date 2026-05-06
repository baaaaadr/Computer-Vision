# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pathlib import Path
import seq1 import find_paper_mask
import seq1 import find_dot
from seq1 import COLOR_RANGES

def find_hand_mask(img, quad_mask):
    """
    Hand detection tuned to observed HSV :
      - Warm skin:  H=0-20,   S=35-140, V=55-135
      - Shadow skin: H=160-180, S=35-105,  V=55-125
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Warm skin pixels
    skin_warm = cv2.inRange(hsv,
        np.array([0,   35, 55], dtype=np.uint8),
        np.array([20, 140, 135], dtype=np.uint8))

    # Shadow/cool skin pixels
    skin_shadow = cv2.inRange(hsv,
        np.array([160, 35, 55], dtype=np.uint8),
        np.array([180, 105, 125], dtype=np.uint8))

    skin_mask = cv2.bitwise_or(skin_warm, skin_shadow)

    # Restrict to the quad (where the overlay is placed)
    skin_on_quad = cv2.bitwise_and(skin_mask, quad_mask)

    # Close large gaps (between fingers)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_on_quad = cv2.morphologyEx(skin_on_quad, cv2.MORPH_CLOSE, kernel_close)
    skin_on_quad = cv2.morphologyEx(skin_on_quad, cv2.MORPH_OPEN,  kernel_open)

    # ── Exclude white AFTER morph ───────
    white_mask = cv2.inRange(hsv,
        np.array([0,  0, 150], dtype=np.uint8),
        np.array([180, 70, 255], dtype=np.uint8))
    skin_on_quad = cv2.bitwise_and(skin_on_quad, cv2.bitwise_not(white_mask))    

# =============================================================================
#     kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     skin_on_quad = cv2.morphologyEx(skin_on_quad, cv2.MORPH_CLOSE, kernel_close)
#     skin_on_quad = cv2.morphologyEx(skin_on_quad, cv2.MORPH_OPEN,  kernel_open)
# =============================================================================

    # Keep only blobs large enough to be a hand
    contours, _ = cv2.findContours(skin_on_quad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_mask = np.zeros_like(skin_on_quad)
    for c in contours:
        if cv2.contourArea(c) > 50:
            cv2.drawContours(hand_mask, [c], -1, 255, thickness=cv2.FILLED)

    return hand_mask



def overlay_image(background_path: str, overlay_path: str, output_path: str):
    """
    Pipeline:
      1. Warp overlay into the paper quad
      2. Detect hand pixels on the paper
      3. Layer: background -> warped_overlay -> hand (from original bg)
    """
    bg  = cv2.imread(background_path)
    ovr = cv2.imread(overlay_path)
    if bg is None:
        raise FileNotFoundError(f"Background not found: {background_path}")
    if ovr is None:
        raise FileNotFoundError(f"Overlay not found: {overlay_path}")

    hsv        = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    paper_mask = find_paper_mask(bg)

    tl = find_dot(hsv, "yellow", paper_mask)
    tr = find_dot(hsv, "red",    paper_mask)
    bl = find_dot(hsv, "blue",   paper_mask)
    br = find_dot(hsv, "green",  paper_mask)

    dst_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    oh, ow = ovr.shape[:2]
    src_pts = np.array(
        [[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]],
        dtype=np.float32,
    )
    
    # ── Warp overlay into quad ────────────────────────────────────────────
    H      = cv2.getPerspectiveTransform(src_pts, dst_pts) #homographie de random points
    warped = cv2.warpPerspective(ovr, H, (bg.shape[1], bg.shape[0])) #apply homographie sur notre cas
    
    quad_mask = np.zeros((bg.shape[0], bg.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(quad_mask, dst_pts.astype(int), 255)

    # Élargir le masque de quelques pixels 
    kernel_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    quad_mask_enlarged = cv2.dilate(quad_mask, kernel, iterations=1)
    
    # ── Detect hand inside the paper region ───────────────────────────────
    hand_mask = find_hand_mask(bg, quad_mask_enlarged)  # restrict to quad, not full image
    
    # ── Composite in 3 layers ─────────────────────────────────────────────
    #   Layer 0 : original background
    #   Layer 1 : warped overlay (quad_mask is on)
    #   Layer 2 : original bg pixels (wherever hand_mask is on)
    quad_3ch = cv2.merge([quad_mask, quad_mask, quad_mask])
    hand_3ch = cv2.merge([hand_mask, hand_mask, hand_mask])
    
    result = np.where(quad_3ch == 255, warped, bg)
    result = np.where(hand_3ch == 255, bg, result)   # hand comes from original bg : appears on top
    
    cv2.imwrite(output_path, result)
    print(f"Saved → {output_path}")
    


# ── Batch processing ──────────────────────────────────────────────────────────
def process_batch(overlay_path: str, backgrounds: list, output_dir: str = "output"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for bg_path in backgrounds:
        stem = Path(bg_path).stem
        out  = str(Path(output_dir) / f"{stem}_result.jpg")
        try:
            overlay_image(bg_path, overlay_path, out)
        except Exception as e:
            print(f"[ERROR] {bg_path}: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import glob

    OVERLAY     = "koala.jpg"
    BACKGROUNDS = glob.glob("seq3/*.png")
    OUTPUT_DIR  = "output"

    process_batch(OVERLAY, BACKGROUNDS, OUTPUT_DIR)
