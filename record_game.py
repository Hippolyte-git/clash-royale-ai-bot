import cv2
import numpy as np
import mss
import time
import os
import pygetwindow as gw
from datetime import datetime
 
def list_windows():
    """Liste toutes les fenêtres disponibles"""
    windows = gw.getAllWindows()
    available = []
    for w in windows:
        if w.title and w.width > 0 and w.height > 0:
            available.append(w)
            print(f"{w._hWnd}: {w.title} ---- Position: {w.left}, {w.top}, Taille: {w.width}x{w.height}")
    return available
 
def record_window(window, output_path, fps=30):
    """
    Enregistre la fenêtre sélectionnée en MP4
    
    Args:
        window: objet fenêtre pygetwindow
        output_path: chemin du fichier de sortie (.mp4)
        fps: images par seconde (défaut 30)
    """
    # Dimensions de la fenêtre
    width = window.width
    height = window.height
 
    # Codec vidéo MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
 
    print(f"\n✅ Enregistrement démarré !")
    print(f"📁 Fichier : {output_path}")
    print(f"📐 Résolution : {width}x{height} @ {fps} FPS")
    print(f"⏹️  Appuie sur CTRL+C pour arrêter\n")
 
    frame_count = 0
    start_time = time.time()
 
    with mss.mss() as sct:
        try:
            while True:
                # Zone de capture = position exacte de la fenêtre
                title_bar_height = 32
                monitor = {
                    "top": window.top + title_bar_height,
                    "left": window.left,
                    "width": width,
                    "height": height -title_bar_height 
                }
 
                # Capture l'image
                screenshot = sct.grab(monitor)
 
                # Convertit en tableau numpy (BGR pour OpenCV)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
 
                # Écrit la frame dans le fichier vidéo
                out.write(frame)
                frame_count += 1
 
                # Affiche les infos toutes les 5 secondes
                elapsed = time.time() - start_time
                if frame_count % (fps * 5) == 0:
                    print(f"⏱️  {elapsed:.0f}s | {frame_count} frames enregistrées")
 
                # Contrôle du FPS
                time.sleep(1.0 / fps)
 
        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            print(f"\n⏹️  Enregistrement arrêté !")
            print(f"⏱️  Durée : {elapsed:.1f} secondes")
            print(f"🎞️  Frames : {frame_count}")
            print(f"💾 Vidéo sauvegardée : {output_path}")
 
    out.release()
 
 
def main():
    print("=== ENREGISTREUR DE PARTIE CLASH ROYALE ===\n")
    print("Fenêtres disponibles :")
    windows = list_windows()
 
    # Choix de la fenêtre
    choice = int(input("\nEntrez le numéro de la fenêtre scrcpy : "))
    selected = None
    for w in windows:
        if w._hWnd == choice:
            selected = w
            break
 
    if selected is None:
        print("❌ Fenêtre introuvable !")
        return
 
    print(f"✅ Fenêtre sélectionnée : {selected.title}")
 
    # Dossier de sortie
    output_dir = "replays"
    os.makedirs(output_dir, exist_ok=True)
 
    # Nom du fichier avec horodatage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"partie_{timestamp}.mp4")
 
    # FPS
    fps_input = input("FPS ? (défaut 30, appuie sur Entrée pour garder) : ").strip()
    fps = int(fps_input) if fps_input else 30
 
    # Lance l'enregistrement
    record_window(selected, output_path, fps)
 
 
if __name__ == "__main__":
    main()
 