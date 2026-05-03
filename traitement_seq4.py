# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:24:13 2026

@author: noahm
"""

import cv2
import numpy as np
from pathlib import Path
import glob

COLOR_RANGES = {
    "yellow": [(15, 80, 100), (35, 255, 255)],
    "red":    [(0, 100, 70), (10, 255, 255)],
    "red2":   [(165, 100, 70), (180, 255, 255)],
    "blue":   [(90, 50, 50), (130, 255, 200)],
    "green":  [(40, 10, 10), (100, 200, 120)], 
}

class RobustTracker:
    def __init__(self, max_jump=60):
        self.prev_pts = None
        self.max_jump = max_jump # Pixels max entre deux frames

    def update(self, detected_dots):
        if self.prev_pts is None:
            if all(d is not None for d in detected_dots):
                self.prev_pts = np.array(detected_dots, dtype=np.float32)
                return self.prev_pts
            return None

        current = np.zeros((4, 2), dtype=np.float32)
        valid_idx = []
        offsets = []

        for i, dot in enumerate(detected_dots):
            if dot is not None:
                dist = np.linalg.norm(np.array(dot) - self.prev_pts[i])
                # Si le point saute trop loin, on suspecte une erreur (main, ombre)
                if dist < self.max_jump:
                    current[i] = dot
                    valid_idx.append(i)
                    offsets.append(current[i] - self.prev_pts[i])
            
        # Extrapolation intelligente
        if len(valid_idx) >= 2:
            avg_offset = np.mean(offsets, axis=0) if offsets else [0, 0]
            for i in range(4):
                if i not in valid_idx:
                    current[i] = self.prev_pts[i] + avg_offset
            
            # Vérification de la forme (éviter l'écrasement en ligne)
            area = cv2.contourArea(current.astype(np.int32))
            if area < 5000: # Si le koala est trop petit, c'est une erreur
                return self.prev_pts
                
            self.prev_pts = current.copy()
            return current
        
        return self.prev_pts

def get_refined_paper_mask(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    # On resserre a et b autour de 128 (gris neutre = papier blanc)
    # La peau est souvent a > 140 ou b > 140.
    mask = (
        (L > 170) & 
        (a > 120) & (a < 136) & 
        (b > 120) & (b < 140)
    ).astype(np.uint8) * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    largest = max(cnts, key=cv2.contourArea)
    paper = np.zeros_like(mask)
    cv2.drawContours(paper, [largest], -1, 255, -1)
    return paper

def find_dot_v2(hsv, color, paper_mask):
    lo, hi = np.array(COLOR_RANGES[color][0]), np.array(COLOR_RANGES[color][1])
    mask = cv2.inRange(hsv, lo, hi)
    if color == "red":
        mask |= cv2.inRange(hsv, np.array(COLOR_RANGES["red2"][0]), np.array(COLOR_RANGES["red2"][1]))
    
    # On cherche UNIQUEMENT dans le papier
    combined = cv2.bitwise_and(mask, paper_mask)
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 15:
            M = cv2.moments(c)
            return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
    return None

def process_robust(overlay_img, input_dir, output_dir, ):
    Path(output_dir).mkdir(exist_ok=True)
    ovr = cv2.imread(overlay_img)
    tracker = RobustTracker(max_jump=80)
    paths = sorted(glob.glob(f"{input_dir}/*.png"))

    for p in paths:
        img = cv2.imread(p)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        paper = get_refined_paper_mask(img)
        
        if paper is None: 
            cv2.imwrite(f"{output_dir}/{Path(p).name}", img)
            continue

        dots = [
            find_dot_v2(hsv, "yellow", paper), # TL
            find_dot_v2(hsv, "red",    paper), # TR
            find_dot_v2(hsv, "green",  paper), # BR
            find_dot_v2(hsv, "blue",   paper)  # BL
        ]
        
        pts = tracker.update(dots)
        
        if pts is not None:
            # --- NOUVEAU GARDE-FOU ---
            # 1. Vérifier si la forme est "convexe" (pas de points croisés)
            is_convex = cv2.isContourConvex(pts.astype(np.int32))
            
            # 2. Vérifier l'aire (si < 10000 pixels, c'est probablement un triangle écrasé)
            area = cv2.contourArea(pts.astype(np.int32))
            
            # 3. Vérifier le ratio d'aspect (facultatif mais utile)
            # Un rectangle de papier ne devient jamais un triangle ultra-fin en réalité.

            if is_convex and area > 8000: 
                # On ne fait le rendu QUE si la géométrie est saine
                h, w = ovr.shape[:2]
                src = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src, pts)
                warped = cv2.warpPerspective(ovr, M, (img.shape[1], img.shape[0]))
                
                ovr_zone = np.zeros_like(paper)
                cv2.fillConvexPoly(ovr_zone, pts.astype(int), 255)
                
                # GESTION MAIN : On ne garde que là où il y a du papier ET l'overlay
                final_mask = cv2.bitwise_and(ovr_zone, paper)
                
                # Adoucir les bords du masque pour éviter l'effet "pixel" sur la main
                final_mask = cv2.GaussianBlur(final_mask, (5,5), 0)
                
                mask_3c = cv2.merge([final_mask]*3) / 255.0
                result = (warped * mask_3c + img * (1 - mask_3c)).astype(np.uint8)
                
                cv2.imwrite(f"{output_dir}/{Path(p).name}", result)
            else:
                # Si la géométrie est suspecte (ex: 116.jpg), on ignore l'overlay
                # On réinitialise le tracker pour qu'il ne reste pas "bloqué" sur l'erreur
                tracker.prev_pts = None 
                cv2.imwrite(f"{output_dir}/{Path(p).name}", img)
        else:
            cv2.imwrite(f"{output_dir}/{Path(p).name}", img)

if __name__ == "__main__":
    # Chemins
    SOURCE_IMAGES = r""
    OVERLAY_IMG   = r""
    OUTPUT_FOLDER = r""
    
    process_robust(OVERLAY_IMG, SOURCE_IMAGES, OUTPUT_FOLDER)