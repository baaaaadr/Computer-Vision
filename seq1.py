# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:04:28 2026

@author: noah maréchal
"""
import cv2
import numpy as np

def detecter_centre_couleur(hsv_img, basse, haute):
    # 1. Créer le masque
    if isinstance(basse, list):
        mask1 = cv2.inRange(hsv_img, basse[0], haute[0])
        mask2 = cv2.inRange(hsv_img, basse[1], haute[1])
        masque = cv2.bitwise_or(mask1, mask2)
    else:
        masque = cv2.inRange(hsv_img, basse, haute)
    
    # 2. Nettoyage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, kernel)
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidats = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 5000:
            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circularite = (4 * np.pi * area) / (peri ** 2)
            
            if circularite > 0.5: 
                candidats.append((cnt, area))
    # 3. Vérification de la forme des contours. (A cause de la main)
    if candidats:
        # On prend la plus grande forme ronde trouvée. Je suis pas sure de celle-ci ça fonctionne pas des masses
        c = max(candidats, key=lambda x: x[1])[0]
        M = cv2.moments(c)
        if M["m00"] > 0:
            return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
    return None

def incruster_avec_orientation(chemin_scene, chemin_logo, chemin_sortie):
    img_scene = cv2.imread(chemin_scene)
    img_logo = cv2.imread(chemin_logo)
    if img_scene is None or img_logo is None: return False

    hsv = cv2.cvtColor(img_scene, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    # --- PLAGES HSV (conseil de l'IA de passer en HSV et pas en RGB) ---
    ranges = {
        "jaune": (np.array([15, 80, 50]), np.array([35, 255, 255])),
        "rouge": ([np.array([0, 50, 20]), np.array([165, 50, 20])], 
                  [np.array([12, 255, 255]), np.array([180, 255, 255])]),
        "bleu":  (np.array([95, 70, 25]), np.array([130, 255, 255])),
        "vert":  (np.array([40, 40, 25]), np.array([90, 255, 150]))
    }

    centres = {}
    for coul, (bas, haut) in ranges.items():
        centres[coul] = detecter_centre_couleur(hsv, bas, haut)

    # On ne traite l'image QUE si on a les 4 points (pour éviter les déformations)
    if None in centres.values():
        print(f"Manquant : {[k for k,v in centres.items() if v is None]}")
        cv2.imwrite(chemin_sortie, img_scene)
        return False

    # --- MAPPING POUR L'HOMOGRAPHIE ---
    pts_dest = np.array([centres["jaune"], centres["rouge"], centres["vert"], centres["bleu"]], dtype=np.float32)
    
    h_l, w_l = img_logo.shape[:2]
    pts_src = np.array([[0, 0], [w_l, 0], [w_l, h_l], [0, h_l]], dtype=np.float32)

    mat_h, _ = cv2.findHomography(pts_src, pts_dest)
    logo_warped = cv2.warpPerspective(img_logo, mat_h, (img_scene.shape[1], img_scene.shape[0]))
    
    # Création du masque pour boucher le trou dans la scène
    mask = np.zeros_like(img_scene)
    cv2.fillConvexPoly(mask, pts_dest.astype(int), (255, 255, 255))
    
    img_final = cv2.bitwise_and(img_scene, cv2.bitwise_not(mask))
    img_final = cv2.add(img_final, logo_warped)

    cv2.imwrite(chemin_sortie, img_final)
    return True