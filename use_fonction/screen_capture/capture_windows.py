"""
Nom du fichier: capture_windows.py
Auteur: Étienne-Théodore PRIN
Date: 2024-02-02

Description:
Ce module est utilisé uniquement si l'os de l'ordinateur est windows
Ce module définie la classe Screen_video_capture qui a pour but de récupérer l'écran du téléphone connecté à l'ordinateur.

Test:
Ce module a été testé sur Windows 11 pro avec CPU intel de 13ème génération et GPU Nvidia série 20super

Classe:
    - Screen_video_capture : classe qui a pour but de récupérer l'écran du téléphone connecté à l'ordinateur

Dépendances:
    - cv2 (openCV)
    - numpy
    - mss
    - pygetwindow
"""

import cv2
import numpy as np
import pygetwindow as gw
import mss
import mss.tools
from use_fonction.configuration.fenetre_def import hauteur_capture,adjusting_x_capture,adjusting_y_capture,adjusting_width_capture,adjusting_height_capture

class Screen_video_capture:
    """
    Classe qui a pour but de récupérer l'écran du téléphone connecté à l'ordinateur en utilisant scrcpy.

    Attributs:
        frame (np.array): dernière image enregistrer de l'écran du téléphone

    Méthodes:
        on_frame(frame): Mais à jour l'attributs frame.
        get_screen(): Renvoie l'attributs frame après avoir été mis en taille désirée
    """
    def __init__(self):
        """
        Créer un objet permettant d'effectuer la capture d'une fenêtre.

        Args:
            None

        Returns:
            None
        """
        self.window = None
        self.hauteur_desirer_capture = hauteur_capture
        self.type_emul = None
        self.facteur_echelle = None
        self.configure_windows()
        self.pygame_name = None

    def configure_windows(self):
        """
        Permet de configurer la fenêtre capturé par le programme

        Args:
            None

        Returns:
            None
        """
        # Récupérer une liste de toutes les fenêtres ouvertes
        all_windows = gw.getAllWindows()

        # Afficher les fenêtres avec un index
        for index, window in enumerate(all_windows):
            if window.title != '':
                print(f"{index}: {window.title} ---- , Position: {window.left}, {window.top}, Taille: {window.width}x{window.height}")
            if window.title == "Clash Royale":
                self.window = gw.getWindowsWithTitle(window.title)[0]
                return None

        # Demander à l'utilisateur de choisir une fenêtre
        try:
            choice = int(input("Entrez le numéro de la fenêtre que vous souhaitez sélectionner : "))
            selected_window = all_windows[choice]
            print(f"Vous avez sélectionné : {selected_window.title}")
            self.window = gw.getWindowsWithTitle(selected_window.title)[0]
            # self.window.activate()
        except ValueError:
            print("Veuillez entrer un nombre entier.")
        except IndexError:
            print("Le numéro entré ne correspond à aucune fenêtre.")

    def get_screen(self):
        """
        Renvoie une image de la fenêtre suivie par notre objet

        Args:
            None

        Returns:
            frame : image de la fenêtre à l'instant t au format couleur OpenCV RGB
        """
        with mss.mss() as sct:        
            # Obtenir les coordonnées de la fenêtre
            x, y, width, height = self.window.left, self.window.top, self.window.width, self.window.height

            # Ajuster les coordonnées pour exclure la barre de titre et les bords
            adjusted_x = x + adjusting_x_capture  # Ajustez si nécessaire
            adjusted_y = y + adjusting_y_capture  # Ajustez pour ignorer la barre de titre
            adjusted_width = width - adjusting_width_capture - adjusting_x_capture  # Ajustez si nécessaire
            adjusted_height = height - adjusting_y_capture - adjusting_height_capture  # Ajustez si nécessaire

            # Validate window position and size
            if adjusted_width <= 0 or adjusted_height <= 0 or adjusted_x < -1000 or adjusted_y < -1000:
                print(f"Warning: Invalid window dimensions detected. Reconfiguring window...")
                self.configure_windows()
                # Try again after reconfiguring
                x, y, width, height = self.window.left, self.window.top, self.window.width, self.window.height
                adjusted_x = x + adjusting_x_capture
                adjusted_y = y + adjusting_y_capture
                adjusted_width = width - adjusting_width_capture - adjusting_x_capture
                adjusted_height = height - adjusting_y_capture - adjusting_height_capture

            monitor = {"top": adjusted_y, "left": adjusted_x, "width": adjusted_width, "height": adjusted_height}

            img = np.array(sct.grab(monitor))  # Capturer l'écran

            largeur = int(img.shape[1])
            hauteur = int(img.shape[0])
            self.facteur_echelle = self.hauteur_desirer_capture/hauteur
            largeur = int(largeur * self.facteur_echelle)
            hauteur = self.hauteur_desirer_capture

            img = cv2.resize(img, (largeur, hauteur))

            # Convertir en format couleur OpenCV
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return frame
    
    def dev_position_to_screen(self, x, y):
        """
        Convert device/game image coordinates to screen coordinates
        
        Args:
            x, y: coordinates in the game image (0 to image_width, 0 to image_height)
        
        Returns:
            [screen_x, screen_y]: coordinates on the actual screen
        """
        # Validate window
        if self.window is None:
            return [100, 100]
        
        # Get actual window dimensions
        window_x = self.window.left
        window_y = self.window.top
        window_w = self.window.width
        window_h = self.window.height
        
        # Account for adjustments made to the capture
        adjusted_left = window_x + adjusting_x_capture
        adjusted_top = window_y + adjusting_y_capture
        adjusted_w = window_w - adjusting_x_capture - adjusting_width_capture
        adjusted_h = window_h - adjusting_y_capture - adjusting_height_capture
        
        # The image dimensions after resizing to hauteur_capture height
        # Height is set to hauteur_capture, width is scaled proportionally
        image_h = self.hauteur_desirer_capture
        image_w = int(adjusted_w * image_h / adjusted_h) if adjusted_h > 0 else adjusted_w
        
        # Convert game coordinates to screen coordinates
        # Normalize to 0-1 range within the game image
        norm_x = x / image_w if image_w > 0 else 0
        norm_y = y / image_h if image_h > 0 else 0
        
        # Clamp to valid range
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
        # Map to screen coordinates
        screen_x = adjusted_left + (norm_x * adjusted_w)
        screen_y = adjusted_top + (norm_y * adjusted_h)
        
        return [screen_x, screen_y]
