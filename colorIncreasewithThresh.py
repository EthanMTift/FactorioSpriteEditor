import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

# ===== Adjustable parameters =====
red_threshold = 120   # how much red the pixel must have
red_boost = 80       # how much to increase red by
green_minus = 30     # how much to decrease green by
blue_minus = 30      # how much to decrease blue by
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

# Create mask where R exceeds threshold AND pixel is not transparent
mask = (r > red_threshold) & (a > 50)

# Make a copy to modify
modified = image.copy()
modified_bgr = modified[:, :, :3].copy()

# ===== FIX: convert to int16 before doing math to avoid uint8 wraparound =====
b_i = modified_bgr[:, :, 0].astype(np.int16)
g_i = modified_bgr[:, :, 1].astype(np.int16)
r_i = modified_bgr[:, :, 2].astype(np.int16)

# Apply the red boost and G/B reduction
r_i[mask] = np.clip(r_i[mask] + red_boost, 0, 255)
g_i[mask] = np.clip(g_i[mask] - green_minus, 0, 255)
b_i[mask] = np.clip(b_i[mask] - blue_minus, 0, 255)

# Convert back to uint8 after all math
modified_bgr = cv2.merge([
    b_i.astype(np.uint8),
    g_i.astype(np.uint8),
    r_i.astype(np.uint8)
])
# ============================================================================

# Merge back alpha
if modified.shape[2] == 4:
    modified = cv2.merge([modified_bgr[:,:,0], modified_bgr[:,:,1], modified_bgr[:,:,2], a])
else:
    modified = modified_bgr

# Save result, preserving alpha
output_path = image_path.rsplit(".", 1)[0] + "_redboosted.png"
cv2.imwrite(output_path, modified)
print(f"Saved boosted image to {output_path}")

# Show result (ignore alpha for display)
display_img = modified[:, :, :3] if modified.shape[2] == 4 else modified
cv2.imshow("Red Boosted Image", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
