import cv2
import numpy as np

img = cv2.imread('silly.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

skin = cv2.bitwise_and(img, img, mask=skin_mask)

cv2.imshow('Original', img)
cv2.imshow('Skin', skin)
if cv2.waitKey(0) & 0xFF == ord('q'):
  cv2.destroyAllWindows()

