import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

# ===== Adjustable parameters =====
green_threshold = 1  # how much higher G must be than R and B
# =================================

app = QApplication(sys.argv)
image_path, _ = QFileDialog.getOpenFileName(None, "Select an image", "", "Images (*.png *.jpg *.jpeg)")
if not image_path:
    print("No image selected.")
    sys.exit()

# Load image including alpha
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if image is None:
    print("Failed to load image.")
    sys.exit()

# Check if image has alpha channel
if image.shape[2] == 4:
    b, g, r, a = cv2.split(image)
else:
    b, g, r = cv2.split(image)
    a = np.ones_like(b) * 255  # fully opaque if no alpha

# Create mask where G is much higher than R and B AND pixel is not transparent
mask = (g > r + green_threshold) & (g > b + green_threshold) & (a > 50)

# Make a copy to modify
swapped = image.copy()

# Swap R and G only on the first 3 channels
swapped_bgr = swapped[:, :, :3].copy()
swapped_bgr[mask] = swapped_bgr[mask][:, [0, 2, 1]]  # swap R and G

# Merge back alpha
if swapped.shape[2] == 4:
    swapped = cv2.merge([swapped_bgr[:,:,0], swapped_bgr[:,:,1], swapped_bgr[:,:,2], a])
else:
    swapped = swapped_bgr

# Save result, preserving alpha
output_path = image_path.rsplit(".", 1)[0] + "_swapped.png"
cv2.imwrite(output_path, swapped)
print(f"Saved swapped image to {output_path}")

# Show result (ignore alpha for display)
display_img = swapped[:, :, :3] if swapped.shape[2] == 4 else swapped
cv2.imshow("Swapped Image", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
