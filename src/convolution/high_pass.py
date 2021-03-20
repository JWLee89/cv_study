import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
Goal of high-pass filters is to
1. Sharpen images
2. Enhance high-frequency parts of an image
"""


def high_pass_filter(image):
    # Good practice to create deep copy of image before working on it
    img_copy = np.copy(image)

    # Convert to gray scale
    # We are looking at patterns of intensity, convert to gray-scale
    # since it captures intensity information compactly
    # Why? because gray-scale images from 0 - 255 represent the amount of light
    # in a given scale I.E. whether it is light (255 - white) or dark (0 - black)
    img_grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


if __name__ == "__main__":
    image = cv2.imread("../../images/cat_image.jpeg")
    # By default, cv2 images are read as BGR instead of RBG
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply high pass filter
    new_image = high_pass_filter(image)