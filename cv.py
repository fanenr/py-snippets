import cv2
import numpy as np

img = cv2.imread("img.jpeg", cv2.IMREAD_COLOR)
h, w = img.shape[:2]

third_h, third_w = h // 3, w // 3
cropped = img[third_h : third_h * 2, third_w : third_w * 2]
cv2.imwrite("cropped.jpeg", cropped)


M = cv2.getRotationMatrix2D((w // 2, h // 2), -45, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))
cv2.imwrite("rotated.jpeg", rotated)


M = np.array([[1, 0, w / 2], [0, 1, h / 2]])
translated = cv2.warpAffine(img, M, (w, h))
cv2.imwrite("translated.jpeg", translated)
