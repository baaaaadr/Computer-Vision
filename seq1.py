# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pathlib import Path

COLOR_RANGES = {
    "yellow": [(5,  100, 150), (30, 255, 255)],  
    "red":    [(0,   120, 80), (10,  255, 255)],
    "red2":   [(170, 120, 80), (180, 255, 255)],
    "blue":   [(100, 60, 60), (141, 135, 105)],
    "green":  [(55,  20,  20), (90,  180, 120)],
}



def find_paper_mask(img):
    # Testing LAB instead of RGB or HSV
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    # detecting WHITE/NEAR-WHITE regions
    mask = (
        (L  > 160) &                
        (a  > 115) & (a < 145) &    
        (b  > 115) & (b < 150)      
    ).astype(np.uint8) * 255

    # Create an elliptical kernel for rounded corners instead of rectangular
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # fill the holes:
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) #fermeture
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel) #ouverture

    # Find all contours (connected white regions) in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contours were found
    if not contours:
        raise RuntimeError("Could not detect the white paper.")
    
    # Find the contour with the largest area (assuming it's the target white paper)
    largest = max(contours, key=cv2.contourArea)
    
    # create mask
    paper_mask = np.zeros_like(mask) 
    cv2.drawContours(paper_mask, [largest], -1, 255, thickness=cv2.FILLED) #draw the largest countour filled with white

    # ── Dilate outward to recover areas hidden under the hand ──────────────
    #before on faisait erosion :
    #erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    #maintenant:
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60)) # thick brush
    paper_mask = cv2.dilate(paper_mask, dilate_kernel, iterations=1)

    return paper_mask



def find_dot(hsv_img, color: str, search_mask: np.ndarray, min_area: int = 30):
    lo   = np.array(COLOR_RANGES[color][0], dtype=np.uint8)
    hi   = np.array(COLOR_RANGES[color][1], dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lo, hi) #mask of the wanted color range

    if color == "red":
        lo2 = np.array(COLOR_RANGES["red2"][0], dtype=np.uint8)
        hi2 = np.array(COLOR_RANGES["red2"][1], dtype=np.uint8)
        mask |= cv2.inRange(hsv_img, lo2, hi2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # ── Two-pass: eroded mask first, full mask as fallback ──────────────────
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    tight_mask   = cv2.erode(search_mask, erode_kernel, iterations=1)

    #attempt 0 : with tight_mash, attempt 1 : with search_mask
    for attempt, region in enumerate([tight_mask, search_mask]):
        candidate = cv2.bitwise_and(mask, region)
        #fill the holes:
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #Only keeps contours larger than min_area (default 30 pixels)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if contours:
            if attempt == 1:
                print(f"  [{color}] found only with full mask (dot near paper edge)")
            largest = max(contours, key=cv2.contourArea)  # Take largest dot
            M = cv2.moments(largest)                    
            cx = int(M["m10"] / M["m00"])     # Center X = sum(x)/sum(area), calculates average X position
            cy = int(M["m01"] / M["m00"])     # Center Y = sum(y)/sum(area), calculates average Y position
            return (cx, cy)

    raise RuntimeError(f"Could not find {color} dot in image.")



def overlay_image(background_path: str, overlay_path: str, output_path: str):
    """
    Warp overlay into the quadrilateral defined by the 4 colored dots.
    Dot -> corner:  yellow=TL, red=TR, blue=BL, green=BR
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

    M      = cv2.getPerspectiveTransform(src_pts, dst_pts) #homographie de random points
    warped = cv2.warpPerspective(ovr, M, (bg.shape[1], bg.shape[0])) #apply homographie sur notre cas

    #créer un mask sur bg pour ovr
    mask    = np.zeros((bg.shape[0], bg.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_pts.astype(int), 255)
    mask_3ch = cv2.merge([mask, mask, mask]) #mask sur les 3 channels de couleurs

    result = np.where(mask_3ch == 255, warped, bg)
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
    BACKGROUNDS = glob.glob("seq1/*.png")
    OUTPUT_DIR  = "output"

    process_batch(OVERLAY, BACKGROUNDS, OUTPUT_DIR)
