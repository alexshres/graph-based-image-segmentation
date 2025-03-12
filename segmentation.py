import numpy as np
import unionfind as uf




class SegmentImage:
    def __init__(self, image):
        self.image = image
        self.size = image.shape[0] * image.shape[1]
        self.S = uf.UnionFind(self.size)
        self.adj_list =  None





