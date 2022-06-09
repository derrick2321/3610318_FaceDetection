# Importing OpenCV
import cv2
# Importing numpy
import numpy as np
original = cv2.imread("photos/flower.jfif")
img = original.copy()
# The function imshow displays
# an image in the specified window.
cv2.imshow("Before", original)
# The function brightens the specified image
Brightness_Matrix = np.ones(original.shape, dtype = "uint8") * 75
#The function print() displays the specified input to the screen
print(Brightness_Matrix)
brightened_image = cv2.add(original, Brightness_Matrix)
# The function imshow displays an image
# in the specified window
cv2.imshow("After",brightened_image)
# The function waitKey waits for
# a key event infinitely  or for
# delay milliseconds, when it is positive.
cv2.waitKey(0)

