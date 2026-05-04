# -*- coding: utf-8 -*-
"""
Created on Sat May  2 01:37:47 2026

@author: Noah Maréchal
"""
import cv2
import numpy as np
from pathlib import Path
import glob
import seq1_FINAL

# --- CONFIGURATION ---
COLOR_RANGES = {
    "red":   [(0, 50, 40), (15, 255, 255)],
    "red2":  [(165, 50, 40), (180, 255, 255)],
}

class PointTracker:
    def __init__(self):
        self.last_pts = None 

    def sort_initially(self, pts):
        """ Trie TL, TR, BR, BL même si la feuille est un peu tournée """
        pts = np.array(pts, dtype="float32")
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-Left
        rect[2] = pts[np.argmax(s)] # Bottom-Right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-Right
        rect[3] = pts[np.argmax(diff)] # Bottom-Left
        return rect

    def update(self, new_pts):
        """ Force l'association par distance """
        if len(new_pts) < 4:
            return self.last_pts # On ne peut rien faire sans 4 points

        # Si c'est le début, on initialise
        if self.last_pts is None:
            self.last_pts = self.sort_initially(new_pts[:4])
            return self.last_pts

        # Calcul des distances entre tous les nouveaux points et les 4 anciens
        matched_pts = np.zeros((4, 2), dtype="float32")
        for i in range(4):
            # Distance entre l'ancien point i et tous les nouveaux points détectés
            dists = np.linalg.norm(new_pts - self.last_pts[i], axis=1)
            matched_pts[i] = new_pts[np.argmin(dists)]
        
        self.last_pts = matched_pts
        return matched_pts

def find_red_dots_advanced(hsv_img, search_mask):
    """ Trouve tous les candidats rouges et les trie par circularité """
    lo, hi = np.array(COLOR_RANGES["red"][0]), np.array(COLOR_RANGES["red"][1])
    lo2, hi2 = np.array(COLOR_RANGES["red2"][0]), np.array(COLOR_RANGES["red2"][1])
    mask = cv2.bitwise_or(cv2.inRange(hsv_img, lo, hi), cv2.inRange(hsv_img, lo2, hi2))
    
    if search_mask is not None:
        mask = cv2.bitwise_and(mask, search_mask)
    
    # Nettoyage pour lisser les ronds
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if 40 < area < 4000:
            peri = cv2.arcLength(c, True)
            circularite = (4 * np.pi * area) / (peri**2) if peri > 0 else 0
            # On stocke le centre
            if circularite > 0.4:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    candidates.append(([cx, cy], circularite))
    
    # On trie par circularité et on garde les meilleurs
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return np.array([x[0] for x in candidates])

def process_sequence(input_dir, overlay_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ovr = cv2.imread(overlay_path)
    img_paths = sorted(glob.glob(f"{input_dir}/*.png") + glob.glob(f"{input_dir}/*.jpg"))
    
    tracker = PointTracker()
    oh, ow = ovr.shape[:2]
    # Coins du logo : TL, TR, BR, BL
    src_pts = np.array([[0, 0], [ow-1, 0], [ow-1, oh-1], [0, oh-1]], dtype=np.float32)

    for path in img_paths:
        bg = cv2.imread(path)
        hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        
        from seq1_FINAL import find_paper_mask
        paper_mask = find_paper_mask(bg)
        
        # On récupère tous les points rouges crédibles
        all_dots = find_red_dots_advanced(hsv, paper_mask)
        
        # Le tracker choisit les 4 cohérents
        dst_pts = tracker.update(all_dots)
        
        if dst_pts is not None:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts.astype(np.float32))
            warped = cv2.warpPerspective(ovr, M, (bg.shape[1], bg.shape[0]))
            
            mask = np.zeros((bg.shape[0], bg.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_pts.astype(int), 255)
            
            result = np.where(cv2.merge([mask]*3) == 255, warped, bg)
            cv2.imwrite(str(Path(output_dir) / Path(path).name), result)
            print(f"Traité : {Path(path).name}")
        else:
            cv2.imwrite(str(Path(output_dir) / Path(path).name), bg)
            print(f"Échec : {Path(path).name}")

# --- LANCEMENT ---
if __name__ == "__main__":
    # Noms des chemins
    SOURCE_IMAGES = r""
    OVERLAY_IMG   = r""
    OUTPUT_FOLDER = r""
    
    process_sequence(SOURCE_IMAGES, OVERLAY_IMG, OUTPUT_FOLDER)
