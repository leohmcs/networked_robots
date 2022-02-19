#!/usr/bin/env python
#coding: utf-8

# TODO
# botar número de robôs como parâmetro do ROS?
# ====================================================================

import rospy
from moveit_msgs.msg import RobotTrajectory

import numpy as np


class BackboneController:
    def __init__(self, net_radius):
        self.net_radius = net_radius
        
        self.trajectory_sub = rospy.Subscriber('robot_trajectory', RobotTrajectory, callback=self.trajectory_callback)

    def trajectory_callback(self, msg):
        msg = RobotTrajectory()
        names = msg.joint_trajectory.joint_names
        trajectory = msg.joint_trajectory.points
        waypoints = self.trajectory_proj(trajectory.positions)

    def trajectory_proj(self, trajectory):
        '''
        Returns each robot path according to the manipulator trajectory
        '''
        print(trajectory)

    
rospy.init_node('backbone_controller')
net_radius = rospy.get_param('network_radius', default=5)

controller = BackboneController(net_radius)
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    rate.sleep()