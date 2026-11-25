"""
Interactive script to extract card and tower vignettes from Figure_1.png
"""

import cv2
import os
from pathlib import Path

# Get directories
script_dir = os.path.dirname(os.path.abspath(__file__))
bot_clash_dir = os.path.dirname(os.path.dirname(script_dir))
image_path = os.path.join(script_dir, 'Figure_1.png')
vignettes_dir = os.path.join(bot_clash_dir, 'use_fonction', 'vignettes')
towers_dir = os.path.join(bot_clash_dir, 'use_fonction', 'towers')

# Create directories if they don't exist
os.makedirs(vignettes_dir, exist_ok=True)
os.makedirs(towers_dir, exist_ok=True)

# Load image
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit(1)

print(f"Image loaded: {image_path}")
print(f"Image shape: {image.shape}")
print()

# Define extraction regions
# Format: (name, output_path, x, y, w, h)
# ADJUST THESE COORDINATES BASED ON YOUR IMAGE!
# x, y = top-left corner
# w, h = width, height
extractions = [
    # Cards (bottom area - adjust y coordinate)
    ("card_1", os.path.join(vignettes_dir, "card_1.jpg"), 20, 380, 60, 70),
    ("card_2", os.path.join(vignettes_dir, "card_2.jpg"), 100, 380, 60, 70),
    ("card_3", os.path.join(vignettes_dir, "card_3.jpg"), 180, 380, 60, 70),
    ("card_4", os.path.join(vignettes_dir, "card_4.jpg"), 260, 380, 60, 70),
    
    # Towers (top area - adjust coordinates as needed)
    ("tower_red", os.path.join(towers_dir, "tower_red.jpg"), 300, 50, 55, 70),
    ("tower_blue", os.path.join(towers_dir, "tower_blue.jpg"), 50, 50, 55, 70),
]

print("=" * 60)
print("ADJUSTMENT INSTRUCTIONS:")
print("=" * 60)
print("Edit the 'extractions' list above with your actual coordinates:")
print("Format: (name, output_path, x, y, w, h)")
print()
print("Example: ('card_1', path, x=100, y=700, w=60, h=70)")
print("  - x, y: top-left corner of the region")
print("  - w, h: width and height of the region")
print()
print("Current image dimensions: width={}, height={}".format(image.shape[1], image.shape[0]))
print("=" * 60)
print()

# Extract all vignettes
extracted_count = 0
for name, output_path, x, y, w, h in extractions:
    # Extract region
    roi = image[y:y+h, x:x+w]
    
    # Save the vignette
    success = cv2.imwrite(output_path, roi)
    
    if success:
        print(f"✓ Extracted {name}: {output_path}")
        extracted_count += 1
    else:
        print(f"✗ Failed to save {name}")

print()
print(f"Extraction complete: {extracted_count}/{len(extractions)} vignettes saved")
print()
print("Next steps:")
print("1. Check the extracted images in:")
print(f"   - {vignettes_dir}")
print(f"   - {towers_dir}")
print("2. If coordinates are wrong, edit the 'extractions' list and run again")
print("3. Run: python initialisation_roi.py (during a live game)")
