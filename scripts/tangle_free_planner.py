#coding: utf-8

import pyvisgraph
import cv2

class TangleFreePlanner:
    def __init__(self, base_pos, goal, occ_grid):
        self.base_pos = base_pos
        self.goal = goal
        self.occ_grid = occ_grid

    def path(self):
        '''
        Returns final path solution
        '''
        pass
