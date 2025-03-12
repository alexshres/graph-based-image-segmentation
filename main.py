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

    img = get_image(image_path)

    test_segment = SegmentImage(img)


    print("Finished testing nbrs from ann_idx.get_nns_by_item")



if __name__ == "__main__":
    main()



    


