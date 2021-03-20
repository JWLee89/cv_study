import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
Goal of high-pass filters is to
1. Sharpen images
2. Enhance high-frequency parts of an image
"""


def visualize_high_pass_filter(image):
    # Good practice to create deep copy of image before working on it
    img_copy = np.copy(image)

    # Convert to gray scale
    # We are looking at patterns of intensity, convert to gray-scale
    # since it captures intensity information compactly
    # Why? because gray-scale images from 0 - 255 represent the amount of light
    # in a given scale I.E. whether it is light (255 - white) or dark (0 - black)
    img_grayscale = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)

    # High pass filters should add to zero.
    # For example, like the 3x3 filter below
    high_pass_kernel = np.array([
        [0, -1, 0],
        [-1, 4, 1],
        [0, -1, 0]
    ])

    # Convolve the image using the high-pass filter
    convolved_image = cv2.filter2D(img_grayscale, -1, high_pass_kernel)

    # let's compare original image with convolved image
    fig, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(10, 5))

    left_axis.set_title("Original image")
    left_axis.imshow(img_grayscale)

    right_axis.set_title("Convolved image")
    right_axis.imgshow(convolved_image)
    fig.show()


if __name__ == "__main__":
    image = cv2.imread("../../images/cat_image.jpeg")
    # By default, cv2 images are read as BGR instead of RBG
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Visualize high pass filter
    visualize_high_pass_filter(image)