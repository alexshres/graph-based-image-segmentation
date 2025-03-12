# FILE: unionfind.py
import numpy as np


class Node:
    def __init__(self, pixel_value):
        self.pixel_value = pixel_value

        # Initially set parent to self
        self.parent = pixel_value

    def __repr__(self):
        return f"Node(id={self.pixel_value}, parent={self.parent})"


class UnionFind:
    def __init__(self, num_pixels):
        self.nodes = [Node(i) for i in range(num_pixels)]
        self.component_sizes = dict() 
        self.internal_diff = dict()

        for i in range(num_pixels):
            # Setting each pixel as a component with size 1
            self.component_sizes[i] = 1

            # Setting each component's internal difference as 0
            self.internal_diff[i] = 0


    def find(self, ele):
        if self.nodes[ele].parent != ele:
            # This optimization will help prevent find function from 
            # having same complexity as DFS for large UnionFinds
            self.nodes[ele].parent = self.find(self.nodes[ele].parent)

        return self.nodes[ele].parent

    
    def union(self, ele1, ele2, int_diff):
        comp1 = self.find(ele1)
        comp2 = self.find(ele2)

        # Always merges smaller component into larger component
        # optimizes tree height to be as small as possible
        if self.component_sizes[comp1] < self.component_sizes[comp2]:
            comp1, comp2 = comp2, comp1

        # Merging the two components
        self.nodes[comp2].parent = comp1
        self.component_sizes[comp1] += self.component_sizes[comp2]

        # Since union is only called when we are merging, we
        # can calculate the new internal difference outside
        # and pass it in as a parameter
        self.internal_diff[comp1] = int_diff

        
        if comp2 in self.component_sizes:
            # slightly faster than using _ = self.component_sizes.pop(comp2)
            del self.component_sizes[comp2]
        
        if comp2 in self.internal_diff:
            del self.internal_diff[comp2]
