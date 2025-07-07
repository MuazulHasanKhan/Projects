import cv2
import numpy as np

# Read images WITH alpha channel (for transparency)
moustache_rgba = cv2.imread('./Data/Train/mustache.png', cv2.IMREAD_UNCHANGED)
glasses_rgba = cv2.imread('./Data/Train/glasses.png', cv2.IMREAD_UNCHANGED)

# Load face image
image = cv2.imread('Data/Test/Before.png')
gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load cascades
eyes_cascade = cv2.CascadeClassifier('./Model/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('./Model/Nose18x15.xml')

# Detect features
eyes = eyes_cascade.detectMultiScale(gray_scale, scaleFactor=1.3, minNeighbors=5)
noses = nose_cascade.detectMultiScale(gray_scale, scaleFactor=1.3, minNeighbors=5)

# Copy for display
image_result = image.copy()

# Overlay function for alpha blending
def overlay_transparent(background, overlay, x, y):
    """Overlay RGBA image onto BGR image"""
    bh, bw = background.shape[:2]
    h, w = overlay.shape[:2]

    if x + w > bw or y + h > bh:
        return background  # Don't overlay if out of bounds

    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    offset = 10
    background[y:y+h, x:x+w] = (1 - mask) * background[y:y+h, x:x+w] + mask * overlay_img
    return background

# Apply glasses
if len(eyes) > 0:
    (x1, y1, w1, h1) = eyes[0]  # Assume first eye pair
    resized_glasses = cv2.resize(glasses_rgba, (w1+20, h1+20))
    image_result = overlay_transparent(image_result, resized_glasses, x1, y1 )

# Apply moustache
if len(noses) > 0:
    (x2, y2, w2, h2) = noses[0]
    resized_moustache = cv2.resize(moustache_rgba, (w2+10, h2+10))
    image_result = overlay_transparent(image_result, resized_moustache, x2 + (w2)//4, y2 + (h2)//4)

cv2.imshow("Snapchat Filter", image_result)

np.savetxt('output', image_result.reshape(-1,3), delimiter=',', header='Channel 1, Channel 2, Channel 3')
cv2.waitKey(0)
cv2.destroyAllWindows()


