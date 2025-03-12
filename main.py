# Alex Shrestha
# Efficient Image Segmentation Implementation
# FILE: main.py 

import os
import numpy as np
import matplotlib.pyplot as plt
import unionfind as uf

from utils import get_image
from segmentation import SegmentImage

def main():
    image_path = "./images/elephant.jpg"

    # img is between 0 and 1 and Gaussian blur with
    # sigma=0.8 has already been applied in the get_image function
    img = get_image(image_path)

    segment = SegmentImage(img)



    print("Finished testing nbrs from ann_idx.get_nns_by_item")



if __name__ == "__main__":
    main()



    


