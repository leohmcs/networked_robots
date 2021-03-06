#!/usr/bin/env python
#coding:utf-8

from octomap_msgs.msg import OctomapWithPose
import numpy as np


class OctomapGenerator:
    def __init__(self):
        self.UNKNOWN = '00'
        self.FREE = '01'
        self.OCCUPIED = '10'
        self.HAS_CHILDREN = '11'
        self.MIN_HEIGHT = 25    # TODO: get net radius from rosparam instead of using a constant

        self.octomap_data = None
        self.max_depth = 16
        self.resolution = 0
        self.root_dim = 0

    def octomap_from_occupancygrid(self, occ_grid, binary=True, occ_thresh=50):
        '''
        Input: nav_msgs/OccupancyGrid message
        Output: octomap_msgs/OctomapWithPose
        '''
        self.resolution = occ_grid.info.resolution
        self.root_dim = np.floor(2**self.max_depth * self.resolution)
        frame_id = occ_grid.header.frame_id
        w, h = occ_grid.info.width , occ_grid.info.height
        self.occ_grid_dim = [w * self.resolution, h * self.resolution]
        occ_grid = np.reshape(occ_grid.data, (h, w))
        occ_grid = self.__fill_missing_cells(occ_grid)

        self.__init_map(occ_grid.shape[0] * self.resolution)

        if binary:
            if occ_thresh < 0 or occ_thresh > 100:
                raise ValueError('Occupancy threshold must be between 0 and 100.')

            self.__generate_binary_octomap(occ_grid, occ_thresh)
        else:
            self.generate_octomap(occ_grid)
            pass
        
        position = self.__map_position()
        octomap = self.__octomap_msg(frame_id, position)
        return octomap

    def __octomap_msg(self, frame_id, position):
        octomap = OctomapWithPose()
        octomap.header.frame_id = frame_id
        # octomap.header.stamp = rospy.Time.now()   not necessary
        octomap.origin.orientation.w = 1.0
        octomap.origin.position.x = -13
        octomap.origin.position.y = -13
        octomap.octomap.binary = True
        octomap.octomap.id = 'OcTree'
        octomap.octomap.resolution = self.__get_resolution()
        octomap.octomap.data = self.octomap_data
        return octomap

    def __init_map(self, l):
        # we store octomap as in int8 array according to OctoMap's serialization implementation
        self.octomap_data = []
        if l > self.root_dim:
            raise ValueError('Occupancy Grid size is greater than OctoMap max size.')
        elif l <= self.root_dim/2:
            self.octomap_data.extend([0, -64])
            largest_voxel_size = self.root_dim/2
            for i in range(self.max_depth - 2, 0, -1):
                self.octomap_data.extend([3, 0])
                largest_voxel_size /= 2
                if l > largest_voxel_size/2:
                    break
        
    # TODO: function to generate non-binary octomap data
    def generate_octomap(self, occ_grid):
        raise NotImplementedError()

    def __generate_binary_octomap(self, occ_grid, occ_thresh=50):
        '''
        Converts an Occupancy Grid into a Binary OctoMap. More precisely, we convert it into an int8 array, which follows the
        OctoMap serialization convention. Therefore, you just need to use OctoMap library to deserialize.
        Input: occ_grid: [m x n] occupancy grid. 
        '''
        
        children = self.__get_children(occ_grid)
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

        parent_data = self.__binary_to_decimal(status[0] + status[1] + status[2] + status[3])
        self.octomap_data.extend([parent_data, parent_data])

        for i in range(8):
            if status[3 - i] == self.HAS_CHILDREN:
                self.__generate_binary_octomap(children[3 - i])

    def __get_children(self, parent):
        '''Returns fours arrays, each one corresponding to a children of parent. Simply splits parent in four quadrants.'''
        dim = parent.shape
        child0 = parent[0:int(np.floor(dim[0]/2)), int(np.ceil(dim[1]/2)):dim[1]]       # first quad
        child1 = parent[0:int(np.floor(dim[0]/2)), 0:int(np.floor(dim[1]/2))]           # second quad
        child2 = parent[int(np.ceil(dim[0]/2)):dim[0], int(np.ceil(dim[1]/2)):dim[1]]   # fourth quad
        child3 = parent[int(np.ceil(dim[0]/2)):dim[0], 0:int(np.floor(dim[1]/2))]       # third quad

        return np.array((child0, child1, child2, child3))

    def __get_resolution(self):
        return self.resolution

    def binarize(self, data, thresh=50):
        data[np.where((data > 0) & (data < thresh))] = 0
        data[np.where(data >= thresh)] = 1
        return data

    def __fill_missing_cells(self, occ_grid):
        dim = occ_grid.shape

        # we need a square
        if dim[0] > dim[1]:
            filling_array = -1 * np.ones((dim[0], dim[0] - dim[1]))
            occ_grid = np.hstack((occ_grid, filling_array))
        elif dim[1] > dim[0]:
            filling_array = -1 * np.ones((dim[1] - dim[0], dim[1]))
            occ_grid = np.vstack((filling_array, occ_grid))

        # the square must have a minimum size
        if occ_grid.shape[0] * self.resolution < self.MIN_HEIGHT:
            occ_grid = self.__fix_min_size(occ_grid) 

        # the square size also must be a power of 2
        if occ_grid.shape[0] % 2 == 1:
            occ_grid = self.__fix_dimensions(occ_grid)

        return occ_grid

    def __fix_dimensions(self, occ_grid):
        ''' The map needs to be a square of size equal to a power of 2. '''
        dim = occ_grid.shape[0]
        power = 1
        while True:
            if dim < 2**power:
                break
            power += 1

        desired = 2**power
        filling_array = -1 * np.ones((dim, desired - dim))
        occ_grid = np.hstack((occ_grid, filling_array))
        filling_array = -1 * np.ones((desired - dim, desired))
        occ_grid = np.vstack((filling_array, occ_grid))
        return occ_grid

    def __fix_min_size(self, occ_grid):
        ''' 
        The OctoMap must have min height of (# of robots)x(network radius) 
        so the manipulator projection doesn't intersect the obstacles in 2D. 
        '''
        dim = occ_grid.shape[0]
        desired = int(self.MIN_HEIGHT / self.resolution)
        filling_array = -1 * np.ones((desired - dim, dim))
        occ_grid = np.vstack((filling_array, occ_grid))
        filling_array = -1 * np.ones((desired, desired - dim))
        occ_grid = np.hstack((occ_grid, filling_array))
        return occ_grid

    def __map_position(self):
        pos = [-self.occ_grid_dim[0]/2, -self.occ_grid_dim[1]/2]
        return pos

    def __log_odds(self, prob):
        return np.log(prob/(1 - prob))
    
    def __binary_to_decimal(self, bin):
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
    from nav_msgs.msg import OccupancyGrid
    import cv2

    occ_map = np.array([[ -1.,  -1.,  -1.,  -1., 100., 100.,  -1.,  -1.],
                        [ -1.,  -1.,  -1.,  -1., 100., 100.,  -1.,  -1.],
                        [ -1.,  -1.,   0.,   0., 100., 100.,  -1.,  -1.],
                        [ -1.,   0.,   100.,   0.,   0., 100.,  -1.,  -1.],
                        [  0.,   0.,   100.,   0.,   0., 100.,  -1.,  -1.],
                        [  0.,   0.,   100.,   0.,   0., 100.,  -1.,  -1.],
                        [  0.,   0.,   100.,   0.,   0.,   0.,  -1.,  -1.],
                        [  0.,   0.,   100.,  -1.,   0.,   0.,  -1.,  -1.]])
    
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.width = occ_map.shape[1]
    msg.info.height = occ_map.shape[0]
    msg.info.resolution = 1.0
    msg.data = occ_map.flatten()

    octomap_gen = OctomapGenerator()
    octomap = octomap_gen.octomap_from_occupancygrid(msg)
    
    print('######################################################################')
    print('Original Occupancy Grid')
    print(occ_map)
    print('\nGenerated OctomapWithPose:\n{}'.format(octomap))
    print('######################################################################\n')