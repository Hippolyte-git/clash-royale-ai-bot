"""
Interactive coordinate finder - Click on image to find coordinates
"""

import cv2
import os

# Get directories
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, 'Figure_1.png')

# Load image
image = cv2.imread(image_path)
display_image = image.copy()

if image is None:
    print(f"Error: Could not load image from {image_path}")
    exit(1)

print(f"Image loaded: {image_path}")
print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} (width x height)")
print()
print("INSTRUCTIONS:")
print("1. Click on the TOP-LEFT corner of the card")
print("2. Click on the BOTTOM-RIGHT corner of the card")
print("3. The coordinates will be printed")
print()

clicks = []

def mouse_callback(event, x, y, flags, param):
    global display_image
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"Click {len(clicks)}: x={x}, y={y}")
        
        # Draw circle at click point
        cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
        
        if len(clicks) == 2:
            # Calculate width and height
            x1, y1 = clicks[0]
            x2, y2 = clicks[1]
            
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            # Ensure correct order
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            
            print()
            print("=" * 50)
            print(f"Corrected coordinates for card_2:")
            print(f"x={x_min}, y={y_min}, w={w}, h={h}")
            print("=" * 50)
            print()
            print("Update the script with these values:")
            print(f'("card_2", os.path.join(vignettes_dir, "card_2.jpg"), {x_min}, {y_min}, {w}, {h}),')
            print()
            
            clicks.clear()

# Create window and set callback
cv2.namedWindow('Card Coordinates Finder')
cv2.setMouseCallback('Card Coordinates Finder', mouse_callback)

print("Click on card_2 region:")
print()

while True:
    cv2.imshow('Card Coordinates Finder', display_image)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset
        display_image = image.copy()
        clicks.clear()
        print("Reset - click again")

cv2.destroyAllWindows()
print("Done!")
