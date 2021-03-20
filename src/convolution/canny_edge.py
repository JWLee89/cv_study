import numpy as np
import cv2
import matplotlib.pyplot as plt
"""
How can we consistently represent thick and thin edges? 
What constitutes as an edge.

Canny edge detection is used widely because it produces accurately
detected edges. It operates as follows

1. Filters out noise using gaussian blur. E.g.
[
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]
Note that gaussian blur is a type of low pass filter

2. Finds the strength and direction of edges using sobel filters. 

In other words, it uses a low pass filter to smoothen out the image and
remove noise signals. Afterwards, it performs edge detection using high pass filters.

3. Applies non-maximum suppression to isolate the strongest edges and thins them to 1px lines.

4. Lastly, applies hysteresis to isolate the best edges.

"""


def get_canny(gray_image, lower, upper):
    img = np.copy(gray_image)
    assert lower < upper
    return cv2.Canny(img, lower, upper)


if __name__ == "__main__":
    image = cv2.imread("../../images/cat_image.jpeg")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    canny_img_wide = get_canny(image_gray, 120, 240)
    canny_img_tight = get_canny(image_gray, 180, 240)

    # Visualize canny
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(30, 10))

    left.set_title("Original")
    middle.set_title("Canny wide")
    right.set_title("Canny tight")

    # Note that wide shows more edges
    # while tight shows less.
    # This is because the cutoff for weak edges is higher at 180 compared to 120
    left.imshow(image_gray, cmap="gray")
    middle.imshow(canny_img_wide, cmap="gray")
    right.imshow(canny_img_tight, cmap="gray")
    fig.show()
