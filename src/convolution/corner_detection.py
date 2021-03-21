import numpy as np
import cv2
import matplotlib.pyplot as plt
"""
We use the Harris corner detection to detect corners.
The definition of a corner is the intersection between two edges. 
Think of the letter "T". at the middle, there is an intersection
between two lines or edges "-" and "|"

Another way to think of corners is a region of large intensity 
in multiple directions.
"""


def corner_detection(gray_image):
    img_cpy = np.copy(gray_image)

    # using cv2
    # 1. blockSize
    # 2. ksize: The size of the sorbel filter in x and y direction
    # 3. k
    # See the following link for more information
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    block_size = 4
    sorbel_filter_size = 5
    k = 0.04
    return cv2.cornerHarris(img_cpy, block_size, sorbel_filter_size, k)


if __name__ == "__main__":
    img = cv2.imread("../../images/img_cat_white.jpeg")
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corner_detected_image = corner_detection(gray_image)

    fig, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(20, 10))

    left_axis.set_title("Original")
    right_axis.set_title("Corner detected")

    left_axis.imshow(gray_image, cmap="gray")
    right_axis.imshow(corner_detected_image, cmap="gray")
    fig.show()
