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
        self.comp_colors = dict()

        for i in range(num_pixels):
            # Setting each pixel as a component with size 1
            self.component_sizes[i] = 1

            # Setting each component's internal difference as 0
            self.internal_diff[i] = 0

            # Each component will have its own specific color for final 
            # segmented image
            self.comp_colors[i] = np.random.randint(0, 256, 3)


    def find(self, ele):
        val = int(ele)
        if self.nodes[val].parent != val: 
            # This optimization will help prevent find function from 
            # having same complexity as DFS for large UnionFinds
            self.nodes[val].parent = int(self.find(self.nodes[val].parent))

        return self.nodes[val].parent

    
    def union(self, ele1, ele2, int_diff):
        comp1 = self.find(int(ele1))
        comp2 = self.find(int(ele2))

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
        if comp2 in self.comp_colors:
            del self.comp_colors[comp2]
