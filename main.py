# Alex Shrestha
# Efficient Image Segmentation Implementation
# FILE: main.py 

import os
import numpy as np
import matplotlib.pyplot as plt
import unionfind as uf
import pprint

from utils import get_image
from segmentation import SegmentImage

def main():
    image_path = "./images/elephant.jpg"

    # img is between 0 and 1 and Gaussian blur with
    # sigma=0.8 has already been applied in the get_image function
    img = get_image(image_path)

    segment = SegmentImage(img, k=300, type="nn")
    segmented_img = segment.segmented_image()

    plt.imshow(segmented_img)
    plt.axis("off")
    plt.title("Segmented Elephant Image")
    plt.show()


    pprint.pprint(segment.S.component_sizes)



if __name__ == "__main__":
    main()



    


