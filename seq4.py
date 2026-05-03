# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:24:13 2026

@author: noahm
"""

import cv2
import numpy as np
import sys
from pathlib import Path

import glob
import os

# Ordre des points : [Haut-Gauche (Jaune), Haut-Droit (Rouge), Bas-Droit (Vert), Bas-Gauche (Bleu)]
COLORS = {
    "yellow": [(15, 50, 100), (45, 255, 255)],
    "red1":   [(0, 50, 40), (15, 255, 255)],
    "red2":   [(165, 50, 40), (180, 255, 255)],
    "blue":   [(90, 30, 30), (130, 255, 255)],
    "green":  [(0, 0, 0), (100, 255, 80)]
   }

def get_paper_mask(img):
    """Version simplifiée et plus robuste pour le débug."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # On prend tout ce qui est plus clair que la table
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY) 
    return mask

def find_corner(hsv, color_name, paper_mask):
    """Trouve le centre d'une pastille de couleur uniquement sur le papier."""
    if color_name == "red":
        mask = cv2.inRange(hsv, COLORS["red1"][0], COLORS["red1"][1])
        mask |= cv2.inRange(hsv, COLORS["red2"][0], COLORS["red2"][1])
    else:
        mask = cv2.inRange(hsv, COLORS[color_name][0], COLORS[color_name][1])
    
    # On croise avec le masque du papier pour ignorer les couleurs sur la main
    mask = cv2.bitwise_and(mask, paper_mask)
    
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 20:
            M = cv2.moments(c)
            return np.array([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
    return None

def solve_missing_point(pts):
    """Calcule le 4ème point manquant par parallélogramme."""
    # pts est une liste [P0, P1, P2, P3] où l'un est None
    idx = [i for i, p in enumerate(pts) if p is None][0]
    
    if idx == 0: # Jaune manque : P0 = P1 + P3 - P2
        pts[0] = pts[1] + pts[3] - pts[2]
    elif idx == 1: # Rouge manque : P1 = P0 + P2 - P3
        pts[1] = pts[0] + pts[2] - pts[3]
    elif idx == 2: # Vert manque  : P2 = P1 + P3 - P0
        pts[2] = pts[1] + pts[3] - pts[0]
    elif idx == 3: # Bleu manque  : P3 = P0 + P2 - P1
        pts[3] = pts[0] + pts[2] - pts[1]
    return pts

def process_frame(img_path, overlay_img):
    img = cv2.imread(img_path)
    ovr = cv2.imread(overlay_img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Masque du papier pour isoler les pastilles et gérer l'occultation par la main
    paper = get_paper_mask(img)
    
    # 2. Détection des 4 coins
    corners = [
        find_corner(hsv, "yellow", paper), # 0
        find_corner(hsv, "red",    paper), # 1
        find_corner(hsv, "green",  paper), # 2
        find_corner(hsv, "blue",   paper)  # 3
    ]
    
    found_count = sum(1 for c in corners if c is not None)
    
    # 3. Sécurité débug demandée
    if found_count < 3:
        print(f" Coins trouvé : {corners}")
        print(f"ERREUR sur {img_path} : seulement {found_count} points détectés. Arrêt du programme.")
        sys.exit()
        
    # 4. Complétion si 3 points sur 4
    if found_count == 3:
        corners = solve_missing_point(corners)
    print(f" Coins trouvé : {corners} sur l'image {img_path}")
    pts_dst = np.array(corners, dtype="float32")
    
    # 5. Incrustation avec Homographie
    h, w = ovr.shape[:2]
    pts_src = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(ovr, M, (img.shape[1], img.shape[0]))
    
    # 6. Masquage final : on n'affiche le koala que là où il y a du papier
    # On crée un polygone rempli pour la zone du koala
    koala_zone = np.zeros_like(paper)
    cv2.fillConvexPoly(koala_zone, pts_dst.astype(int), 255)
    
    # Le masque final est l'intersection de la zone du koala et du papier visible
    final_mask = cv2.bitwise_and(koala_zone, paper)
    final_mask_3c = cv2.merge([final_mask]*3) / 255.0
    
    # Fusion
    result = (warped * final_mask_3c + img * (1 - final_mask_3c)).astype(np.uint8)
    return result
def process_complete(img_folder, output_folder, overlay_img):
    """
    Parcourt toutes les images d'un dossier, applique l'incrustation
    et sauvegarde les résultats dans un dossier de sortie.
    """
    # 1. Création du dossier de sortie s'il n'existe pas
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Récupération de la liste des images (.jpg, .png, etc.)
    # On trie par nom pour garder l'ordre de la séquence
    extensions = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(img_folder, ext)))
    
    image_paths.sort()

    if not image_paths:
        print(f"Aucune image trouvée dans : {img_folder}")
        return

    print(f"Début du traitement : {len(image_paths)} images à traiter.")

    # 3. Boucle de traitement
    for img_path in image_paths:
        file_name = Path(img_path).name
        
        try:
            # Appel de la fonction de traitement unitaire
            result = process_frame(img_path, overlay_img)
            
            # Sauvegarde du résultat
            save_path = output_path / file_name
            cv2.imwrite(str(save_path), result)
            print(f"OK : {file_name}")
            
        except SystemExit:
            # Si sys.exit() est appelé par process_frame (moins de 3 points)
            print(f"Arrêt critique sur l'image {file_name}. Vérifiez vos pastilles.")
            break
        except Exception as e:
            print(f"Erreur inattendue sur {file_name} : {e}")
            break

    print("Traitement terminé.")
if __name__ == "__main__":
    # Chemins
    SOURCE_IMAGES = r""
    OVERLAY_IMG   = r""
    OUTPUT_FOLDER = r""
    
    process_complete(SOURCE_IMAGES, OUTPUT_FOLDER, OVERLAY_IMG)
