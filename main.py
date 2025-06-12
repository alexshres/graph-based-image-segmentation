# Alex Shrestha
# FILE: main.py 

import sys
import cv2
import matplotlib.pyplot as plt

from utils import get_image
from segmentation import SegmentImage


def main(img_file_path, k=200):

    img = get_image(img_file_path)


    sigma = 0.8 # based on the paper


    eff_img = cv2.GaussianBlur(img, (0, 0), sigma)



    segment = SegmentImage(eff_img, k=200, type="nn")
    segmented_img = segment.segmented_image()


    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(eff_img)
    ax1.axis("off")
    ax1.set_title("Original Image")

    ax2.imshow(segmented_img)
    ax2.axis("off")
    ax2.set_title("Segmented Image")

    plt.show()


if __name__ == "__main__":
    image_file = sys.argv[1]

    if len(sys.argv) == 3:
        k = sys.argv[2]
        main(image_file, int(k))
    else:
        main(image_file)

