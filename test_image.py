# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:56:11 2026

@author: noah maréchal
"""
# Ici c'est que de l'IA cela sert à sortir les vidéos directement


import cv2
import os

def extraire_images_video(chemin_video, dossier_destination):
    """
    Extrait toutes les images d'une vidéo et les enregistre dans un dossier.
    
    Args:
        chemin_video (str): Chemin complet vers le fichier .mp4
        dossier_destination (str): Dossier où enregistrer les images .png
    """
    
    # 1. Créer le dossier de destination
    if not os.path.exists(dossier_destination):
        os.makedirs(dossier_destination)
        print(f"Dossier créé : {dossier_destination}")

    # 2. Charger la vidéo
    video = cv2.VideoCapture(chemin_video)
    
    if not video.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo. Vérifiez le chemin.")
        return

    succes, image = video.read()
    compteur = 0

    print("Extraction en cours...")

    # 3. Boucle de lecture de la vidéo
    while succes:
        nom_fichier = os.path.join(dossier_destination, f"{compteur:03d}.png")
        
        cv2.imwrite(nom_fichier, image)
        
        succes, image = video.read()
        compteur += 1

    # 4. Libérer les ressources
    video.release()
    print(f"Terminé ! {compteur} images ont été extraites dans : {dossier_destination}")

def creer_video_images(dossier_images, chemin_video_sortie, fps=10):
    """
    Crée une vidéo à partir de toutes les images .png d'un dossier.
    
    Args:
        dossier_images (str): Dossier contenant les images (ex: 001.png, 002.png)
        chemin_video_sortie (str): Chemin complet du fichier de sortie (ex: output.mp4)
        fps (int): Nombre d'images par seconde (équivalent au -framerate de ffmpeg)
    """
    
    # 1. Récupérer et trier la liste des images
    images = [img for img in os.listdir(dossier_images) if img.endswith(".png")]
    images.sort() 

    if not images:
        print("Erreur : Aucune image .png trouvée dans le dossier.")
        return

    # 2. Lire la première image pour obtenir les dimensions (largeur, hauteur)
    premiere_image_path = os.path.join(dossier_images, images[0])
    frame = cv2.imread(premiere_image_path)
    hauteur, largeur, couches = frame.shape

    # 3. Définir le codec et créer l'objet VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(chemin_video_sortie, fourcc, fps, (largeur, hauteur))

    print(f"Création de la vidéo à {fps} FPS...")

    # 4. Ajouter chaque image à la vidéo
    for nom_img in images:
        chemin_img = os.path.join(dossier_images, nom_img)
        img = cv2.imread(chemin_img)
        video.write(img)

    # 5. Libérer la vidéo
    video.release()
    print(f"Vidéo enregistrée avec succès : {chemin_video_sortie}")