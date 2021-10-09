#!/usr/bin/env python
#coding: utf-8

import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus
from interconnected_robots.msg import *
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

from utils.functions import *

class Navigator:
    def __init__(self):
        self.client = actionlib.SimpleActionClient('reactive_planner', ReactivePlannerAction)

    def reactive(self):
        # testando, depois mudo para o parâmetro
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = -6.9000e+00
        goal.pose.position.y = -2.5250e+00
        
        if self.client.get_state() != GoalStatus.ACTIVE:
            self.client.wait_for_server()
            self.client.send_goal(ReactivePlannerGoal(goal=goal))
            self.client.wait_for_result()

        return self.client.get_result()

node_name = 'navigator'
rospy.init_node(node_name)
rospy.loginfo(node_name + ' inicializado com sucesso.')

node = Navigator()

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    result = node.reactive()
# print(result)
    rate.sleep()


