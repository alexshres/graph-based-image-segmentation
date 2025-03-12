# Alex Shrestha
# Efficient Image Segmentation Implementation
# FILE: main.py 

import os
import numpy as np
import matplotlib.pyplot as plt

import unionfind as uf
# import utils


def main():
    test_node = uf.Node(5)
    print(f"{test_node=}")

    test_unionfind = uf.UnionFind(5)

    print(f"{test_unionfind.elements=}\n")
    print(f"{test_unionfind.component_size=}\n")
    print(f"{test_unionfind.internal_diff=}\n")



if __name__ == "__main__":
    main()



    


