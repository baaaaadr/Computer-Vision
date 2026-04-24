# -*- coding: utf-8 -*-
"""
Created on Wed Apr 4 19:20:44 2026

@author: noah maréchal
"""

import os
import cv2
from seq1 import incruster_avec_orientation

def generer_sequence_traitee():
    # --- CONFIGURATION DES CHEMINS (Les noms des chemins sont en durs) ---
    dossier_source = r""
    chemin_logo = r""
    dossier_sortie = r""
    nom_video_finale = "seq1_traite.mp4"
    
    # Création du dossier de sortie s'il n'existe pas
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
        print(f"Dossier créé : {dossier_sortie}")

    images_traitees_paths = []

    # --- TRAITEMENT IMAGE PAR IMAGE ---
    print("Début du traitement des images (000 à 200)...")
    
    for i in range(200):
        nom_img = f"{i:03d}.png"
        chemin_scene = os.path.join(dossier_source, nom_img)
        chemin_sortie_img = os.path.join(dossier_sortie, f"traite_{nom_img}")

        if os.path.exists(chemin_scene):
            # Appel de votre fonction existante
            succes = incruster_avec_orientation(chemin_scene, chemin_logo, chemin_sortie_img)
            
            if succes:
                images_traitees_paths.append(chemin_sortie_img)
                if i % 20 == 0:
                    print(f"Image {nom_img} traitée...")
            else:
                print(f"Échec du traitement pour {nom_img} (pastilles non trouvées ?)")
        else:
            print(f"Fichier introuvable : {chemin_scene}")

    # --- CRÉATION DE LA VIDÉO ---
    if not images_traitees_paths:
        print("Erreur : Aucune image n'a été traitée. Vidéo non générée.")
        return

    print("Création de la vidéo finale...")
    
    # Lecture de la première image pour les dimensions
    premiere_img = cv2.imread(images_traitees_paths[0])
    hauteur, largeur, _ = premiere_img.shape
    
    chemin_video_finale = os.path.join(dossier_sortie, nom_video_finale)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(chemin_video_finale, fourcc, 10, (largeur, hauteur))

    for chemin in images_traitees_paths:
        img = cv2.imread(chemin)
        video_writer.write(img)

    video_writer.release()
    print(f"Terminé ! Vidéo enregistrée sous : {chemin_video_finale}")

if __name__ == "__main__":
    generer_sequence_traitee()