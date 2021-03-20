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
    gray_image = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)

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

    return cv2.filter2D(gray_image, -1, low_pass_filter)


if __name__ == "__main__":
    image = cv2.imread("../../images/cat_image.jpeg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Try fiddling around with this value
    weight = 2
    noise_removed = remove_noise(image, weight)

    # Visualize images before and after removing noise
    fig, (left_axis, right_axis) = plt.subplots(1, 2, figsize=(20, 10))

    left_axis.set_title("Original image")
    left_axis.imshow(image_rgb, cmap='gray')

    right_axis.set_title("Noise removed (with low pass filter)")
    right_axis.imshow(noise_removed, cmap='gray')
    fig.show()