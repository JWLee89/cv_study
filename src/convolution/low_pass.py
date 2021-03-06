import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
Main idea: Low pass filters block high-frequency parts of an image. 
In other words, it has blurring effect on images.

Mathematically, we are taking the average of the center pixel and its surrounding
neighbors.
"""


def remove_noise(image, weight):
    img_copy = np.copy(image)

    # Apply low pass filter to remove image.
    # The way this works is that we remove low-intensity pixels from the image
    # to smoothen out the image
    # In the example above, we weigh all neighboring pixels the same
    # IMPORTANT: The sum of all the weights should add up to one
    low_pass_filter = 1/9 * np.array([
        [weight, weight, weight],
        [weight, weight, weight],
        [weight, weight, weight]
    ])

    return cv2.filter2D(img_copy, -1, low_pass_filter)


def gaussian_blur(image):
    """
    Might be the most commonly used low-pass filter.
    It is useful for preserving edges and
    blocking high-frequency parts of an image.

    Once again, the values are positive and normalized, meaning the weighted sum of
    values in the filter add up to one.
    """
    img_copy = np.copy(image)

    # Sums up to 16
    # Weighted sum of pixels based on distance from center
    # Gives most weight to center pixel and
    # proportionally weighs neighboring pixels based
    # on distance from center pixed
    gaussian_filter = 1/16 * np.array([
        [1, 2, 1],      # 4
        [2, 4, 2],      # 8
        [1, 2, 1]       # 4
    ])

    return cv2.filter2D(img_copy, -1, gaussian_filter)


if __name__ == "__main__":
    image = cv2.imread("../../images/cat_image.jpeg")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Try fiddling around with this value
    weight = 2
    noise_removed = remove_noise(image, weight)

    # Visualize images before and after removing noise
    fig, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(20, 10))

    left_axis.set_title("Original image")
    left_axis.imshow(image_gray, cmap='gray')

    right_axis.set_title("Noise removed (with low pass filter)")
    right_axis.imshow(noise_removed, cmap='gray')
    fig.show()

    # Visualize gaussian blur
    fig, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(20, 10))

    left_axis.set_title("Original image")
    left_axis.imshow(image_gray, cmap='gray')

    right_axis.set_title("Gaussian blur")
    right_axis.imshow(gaussian_blur(image_gray), cmap='gray')
    fig.show()

    # Apply sobel vertical to images
    sobel_vertical = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    fig, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(20, 10))

    left_axis.set_title("Original image (sobel filtered)")
    left_axis.imshow(cv2.filter2D(image_gray, -1, sobel_vertical), cmap='gray')

    right_axis.set_title("Gaussian blur (sobel filtered)")
    right_axis.imshow(cv2.filter2D(gaussian_blur(image_gray), -1, sobel_vertical), cmap='gray')
    fig.show()


