#!/usr/bin/env python
#coding:utf-8

import numpy as np


class OctomapGenerator:
    def __init__(self):
        self.UNKNOWN = '00'
        self.FREE = '01'
        self.OCCUPIED = '10'
        self.HAS_CHILDREN = '11'
        self.MIN_HEIGHT = 25    # TODO: get net radius from rosparam instead of using a constant

        self.octomap = None
        self.root_dim = 0
        self.max_depth = 1
    
    def binarize(self, data, thresh=50):
        data[np.where((data > 0) & (data < thresh))] = 0
        data[np.where(data >= thresh)] = 1
        return data

    # TODO
    def fill_missing_cells(self, occ_grid):
        ''' 
        Because the OctoMap has a set dimension, we may need to fill the Occupancy Grid 
        with unknown cells to match the OctoMap dimensions.
        '''
        return occ_grid

    def octomap_from_occupancygrid(self, occ_grid, resolution, binary=True, occ_thresh=50):
        '''
        Input: occupancy grid [m x n]
        '''
        self.init_map()
        occ_grid = self.fill_missing_cells(occ_grid)
        dim = occ_grid.shape[0] * resolution
        self.root_dim = np.max([np.max(dim), self.MIN_HEIGHT])

        if binary:
            if occ_thresh < 0 or occ_thresh > 100:
                raise ValueError('Occupancy threshold must be between 0 and 100.')

            self.generate_binary_octomap(occ_grid, occ_thresh)
        else:
            self.generate_octomap(occ_grid)
            pass

        return True

    def init_map(self):
        # we store octomap as in int8 array according to OctoMap's serialization implementation
        self.octomap = []

    def generate_octomap(self, occ_grid):
        raise NotImplementedError()

    def generate_binary_octomap(self, occ_grid, occ_thresh=50):
        '''
        Converts an Occupancy Grid into a Binary OctoMap. More precisely, we convert it into an int8 array, which follows the
        OctoMap serialization convention. Therefore, you just need to use OctoMap library to deserialize.
        Input: occ_grid: [m x n] occupancy grid. 
        '''
        
        children = self.get_children(occ_grid)
        status = []
        for i in range(4):
            child = children[i]
            if np.all(child >= occ_thresh):
                status.append(self.OCCUPIED)
            elif np.all(child == -1):
                status.append(self.UNKNOWN)
            elif np.all((child > -1) & (child < occ_thresh)):
                status.append(self.FREE)
            else:
                status.append(self.HAS_CHILDREN)

        if np.any(np.array(status) == self.HAS_CHILDREN):
            self.max_depth += 1

        parent_data = self.binary_to_decimal(status[0] + status[1] + status[2] + status[3])
        self.octomap.extend([parent_data, parent_data])
        
        print('####### Mapa completo #######')
        print(occ_grid)
        print('{} {} {} {} = {}'.format(status[0], status[1], status[2], status[3], parent_data))
        print('\noctomap: {}'.format(self.octomap))
        print('#############################\n')

        for i in range(8):
            if status[3 - i] == self.HAS_CHILDREN:
                self.generate_binary_octomap(children[3 - i])

    def get_children(self, parent):
        '''Returns fours arrays, each one corresponding to a children of parent. Simply splits parent in four quadrants.'''
        dim = parent.shape
        child0 = parent[0:int(np.floor(dim[0]/2)), int(np.ceil(dim[1]/2)):dim[1]]       # first quad
        child1 = parent[0:int(np.floor(dim[0]/2)), 0:int(np.floor(dim[1]/2))]           # second quad
        child2 = parent[int(np.ceil(dim[0]/2)):dim[0], int(np.ceil(dim[1]/2)):dim[1]]   # fourth quad
        child3 = parent[int(np.ceil(dim[0]/2)):dim[0], 0:int(np.floor(dim[1]/2))]       # third quad

        return np.array((child0, child1, child2, child3))

    def get_resolution(self):
        return self.root_dim / (2 ** self.max_depth)    # root_dim = 2**max_depth * resolution

    def log_odds(self, prob):
        return np.log(prob/(1 - prob))
    
    def binary_to_decimal(self, bin):
        ''' Converts a binary input string into a decimal int following two complement convention. '''
        dec = 0
        if bin[0] == '1':
            for i in range(len(bin) - 1):
                dec += 2**i * (1 - int(bin[-1 - i]))
            dec += 1
            dec *= -1
        else:
            for i in range(len(bin)):
                dec += 2**i * int(bin[-1 - i])

        return dec   


if __name__ == '__main__':
    # Example
    occ_map = np.array([[ -1.,  -1.,  -1.,  -1., 100., 100.,  -1.,  -1.],
                        [ -1.,  -1.,  -1.,  -1., 100., 100.,  -1.,  -1.],
                        [ -1.,  -1.,   0.,   0., 100., 100.,  -1.,  -1.],
                        [ -1., 100.,   0.,   0.,   0., 100.,  -1.,  -1.],
                        [100., 100.,   0.,   0.,   0., 100.,  -1.,  -1.],
                        [100., 100.,   0.,   0.,   0., 100.,  -1.,  -1.],
                        [100., 100.,   0.,   0.,   0.,   0.,  -1.,  -1.],
                        [100.,   0.,   0.,  -1.,   0.,   0.,  -1.,  -1.]])
    
    octomap_gen = OctomapGenerator()
    octomap_gen.octomap_from_occupancygrid(occ_map, 0.5)
    
    print('######################################################################')
    print('Original Occupancy Grid')
    print(occ_map)
    print('\nGenerated Octomap Data:\n{}'.format(octomap_gen.octomap))
    print('######################################################################\n')