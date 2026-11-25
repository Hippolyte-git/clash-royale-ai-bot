"""
Nom du fichier: screen_analyse.py
Auteur: Étienne-Théodore PRIN
Date: 2024-02-02

Description:
Ce module définie la classe Screen_analyse qui a pour but d'analyser une image du jeu Clash Royal pour en donner l'état.

Classe:
    - Screen_analyse : classe qui a pour but d'analyser une image du jeu Clash Royal pour en donner l'état.

Dépendances:
    - torch
    - torchvision
    - nécessite la création du CNN vgg10 (initialisable avec ai_creation.ia_ingame.ia_ingame_learning.py)
    - ultralytics
    - nécessite la création d'un yolo avec le code train_yolo.py et d'une base de donnée
    - open CV (cv2)
    - os
    - use_fonction.configuration.p2p (initialisation de variable)
"""

import torch
from torchvision import transforms
from ai_creation.ia_ingame.ia_ingame_model import VGG10
from use_fonction.configuration.p2p import chemin_vgg10,chemin_yolo,classes_yolo,ROI_cartes,chemin_dossier_vignettes,ROI_elixir,ROI_tower,chemin_dossier_tours
from ultralytics import YOLO
import cv2
import os
import numpy as np

class Screen_analyse:
    """
    Classe qui a pour but d'analyser une image du jeu Clash Royal pour en donner l'état.

    Méthodes:
        public:
        get_state(img): Renvoie l'état de l'image.
        private:
        in_game(img): indique si l'image correspond à l'état en jeu ou dans le menu.
        get_cartes(img) : indique les cartes disponibles dans la partie
        get_elixir(img) : indique l'elixir disponible dans la partie
        get_tower_state(img) : indique si les tours sont détruite ou non dans la partie
    """
    def __init__(self, yolo_conf=0.35):
        """
        Créer un objet permettant d'analyser une image du jeu clash royal.

        Args:
            yolo_conf (float): seuil de confiance minimal pour les détections YOLOv8.

        Returns:
            None
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.AI_in_game = VGG10(num_classes=2).to(self.device)
        self.AI_in_game.load_state_dict(torch.load(chemin_vgg10, map_location=self.device))
        self.AI_in_game.eval()
        self.AI_process_in_game = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        self.yolo = YOLO(chemin_yolo)
        self.yolo_conf = yolo_conf
        self.classes = self.yolo.names if hasattr(self.yolo, 'names') and self.yolo.names else classes_yolo
        self.cartes_roi = ROI_cartes
        self.dossier_vignettes = chemin_dossier_vignettes
        self.classes_cartes = []
        self.elixir_roi = ROI_elixir
        self.tower_roi = ROI_tower
        self.dossier_image_tower = chemin_dossier_tours

        # Cache templates in memory to avoid disk I/O every frame
        self._vignette_templates = self._load_templates(self.dossier_vignettes)
        self._tower_templates = self._load_templates(self.dossier_image_tower)

    @staticmethod
    def _load_templates(directory):
        templates = {}
        if os.path.isdir(directory):
            for fichier in os.listdir(directory):
                chemin_image = os.path.join(directory, fichier)
                image = cv2.imread(chemin_image)
                if image is not None:
                    templates[fichier] = image
        return templates
    
    def get_state(self, img):
        """
        Analyse une image du jeu clash royal.

        Args:
            img : image du jeu clash royal

        Returns:
            state : état du jeu selon la forme : 
                [True,[0,"elixir",5],4*[0,"carte",carte.str],x*[1,[pos],"nom_objet",[w,h]]] si en partie 
                ou [False] si dans les menus
            la présence d'un 1 en début de chaque éléments indique la nécessité de rajouter la bouding boxes lors de l'affichage sur l'UI
        """
        if not self.in_game(img):
            return [False]
        state = [True]

        # seuillage pour trouver la quantité d'elixir
        elixir=self.get_elixir(img)
        state.append([0,"elixir :",elixir])

        # analyse tours
        towers_state = self.get_tower_state(img)
        state += towers_state

        #pix to pix pour trouver les cartes
        cartes=self.get_cartes(img)
        for i in range(len(cartes)):
            state.append([0,"carte "+str(i),cartes[i]])
        # yolo pour trouver les unités
        results = self.yolo(img, conf=self.yolo_conf, verbose=False)
        for result in results:
            boxes = result.boxes.xywh.cpu().numpy()
            for u, i in enumerate(boxes):
                cls_id = int(result.boxes.cls[u])
                label = self.classes[cls_id] if cls_id in self.classes or (isinstance(self.classes, list) and cls_id < len(self.classes)) else str(cls_id)
                state.append([1, [int(i[0]), int(i[1])], str(label), [int(i[2]), int(i[3])]])
        return state
    
    def in_game(self, img):
        """
        Analyse une image du jeu clash royal pour savoir si on est en jeu ou dans les menus.

        Args:
            img : image du jeu clash royal

        Returns:
            state : True : en partie
                    False : dans les menus
        """
        input_image = self.AI_process_in_game(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.AI_in_game(input_image)
            _, predicted_class = torch.max(prediction, 1)
        return predicted_class.item() == 0
    
    def get_cartes(self, img):
        """
        Analyse une image du jeu clash royal en partie pour connaitre les cartes disponibles.

        Args:
            img : image du jeu clash royal en partie

        Returns:
            cartes : ["nom_cartes_1","nom_cartes_2","nom_cartes_3","nom_cartes_4"]
        """
        cartes = [0, 0, 0, 0]
        seuil = 0.15
        for i in range(len(cartes)):
            x, y, w, h = self.cartes_roi[i]
            roi = img[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            max_seuil = 0
            for fichier, image in self._vignette_templates.items():
                resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
                resultat = cv2.matchTemplate(resized, roi, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(resultat)
                if max_val >= seuil and max_val > max_seuil:
                    max_seuil = max_val
                    cartes[i] = fichier
        return cartes
    
    def get_elixir(self, img):
        """
        Analyse une image du jeu clash royal en partie pour connaitre l'elixir disponible.

        Args:
            img : image du jeu clash royal en partie

        Returns:
            elixir (int) : l'elixir disponible un entier en 0 et 10 inclus
        """
        x, y, w, h = self.elixir_roi
        if w <= 0:
            return 0
        elixir_bar = img[y:y+h, x:x+w][0]
        occurrences = len([px for px in elixir_bar if (px[0] > 150 and px[2] > 150)])
        return max(0, min(10, int(((occurrences + 20) / w) * 10)))
    
    def get_tower_state(self, img):
        """
        Analyse une image du jeu clash royal en partie pour connaitre l'état des tours.

        Args:
            img : image du jeu clash royal en partie

        Returns:
            state : renvoie une liste de 4 liste de la forme [1,[pos_x,pos_y],"alive/destoyed tower",[h,w]]
        """
        state = []
        seuil = 0.25
        for x, y, w, h in self.tower_roi:
            roi = img[y:y+h, x:x+w]
            part_state = [1, [x + w // 2, y + h // 2], "destroyed tower", [w, h]]
            if roi.size == 0:
                state.append(part_state)
                continue
            max_seuil = 0
            for fichier, image in self._tower_templates.items():
                try:
                    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
                    resultat = cv2.matchTemplate(resized, roi, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(resultat)
                    if max_val >= seuil and max_val > max_seuil:
                        max_seuil = max_val
                        part_state = [1, [x + w // 2, y + h // 2], "alive tower", [w, h]]
                except cv2.error:
                    continue
            state.append(part_state)
        return state
