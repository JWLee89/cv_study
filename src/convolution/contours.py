import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
Learning to detect contours is important for
tasks such as object detection
"""


def detect_contours(gray_image):
    img_cpy = np.copy(gray_image)

    # Create binary threshold, but swap black for white
    # meaning 1 becomes black and 0 becomes white
    threshold = 230

    # Value assigned to pixels exceeding threshold value
    value_assigned = 255

    # retval: the threshold that was used
    # in this case, it would be 220
    # threshold_image: The thresholded image (black and white image)
    retval, thresholded_image = cv2.threshold(img_cpy,
                          threshold, value_assigned, cv2.THRESH_BINARY_INV)

    # Find contours from binary image
    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours
    # -1 means draw all contours
    # (0, 255, 0) represents the color for drawing
    # 3 is the thickness of the contour
    color = (0, 255, 0)
    thickness = 3
    contour_image = cv2.drawContours(img_cpy, contours, -1, color, thickness)

    angles = orientations(contours)
    print(f'Angles of each contour (in degrees): {angles}')

    return thresholded_image, contour_image


def orientations(contours):
    """
    Returns the orientations of a list of contours .
    The list should be in the same order as the contours
    :param contours: a list of contours
    :return: angles, the orientations of the contours
    """

    # Create an empty list to store the angles in
    angles = []
    for contour in contours:
        print(contour)
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        angles.append(angle)

    return angles


if __name__ == "__main__":
    img = cv2.imread("../../images/hand.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresholded_image, contour_image = detect_contours(gray_img)

    fig, (left_axis, middle_axis, right_axis) = plt.subplots(1, 3, figsize=(20, 10))

    left_axis.set_title("Original")
    middle_axis.set_title("Thresholded image inverse")
    right_axis.set_title("Contours")

    left_axis.imshow(gray_img, cmap="gray")
    middle_axis.imshow(thresholded_image, cmap="gray")
    right_axis.imshow(contour_image, cmap="gray")

    fig.show()
