import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

# ===== Adjustable parameters =====
green_threshold = 40  # how much higher G must be than R and B
# =================================

app = QApplication(sys.argv)
image_path, _ = QFileDialog.getOpenFileName(None, "Select an image", "", "Images (*.png *.jpg *.jpeg)")
if not image_path:
    print("No image selected.")
    sys.exit()

# Load image (OpenCV loads in BGR order)
image = cv2.imread(image_path)
if image is None:
    print("Failed to load image.")
    sys.exit()

# Split channels
b, g, r = cv2.split(image)

# Create mask where G is much higher than both R and B
mask = (g > r + green_threshold) & (g > b + green_threshold) 

# Make a copy to modify
swapped = image.copy()

# Swap R and G only where mask is True
swapped[mask] = swapped[mask][:, [0, 2, 1]]

# Save result
output_path = image_path.rsplit(".", 1)[0] + "_swapped.png"
cv2.imwrite(output_path, swapped)
print(f"Saved swapped image to {output_path}")

# Show result
cv2.imshow("Swapped Image", swapped)
cv2.waitKey(0)
cv2.destroyAllWindows()
