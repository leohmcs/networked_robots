#!/usr/bin/env python
#coding: utf-8

import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, Twist
from interconnected_robots.msg import *
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan

from nav_msgs.srv import GetPlan

import numpy as np

from utils.functions import *

class ReactiveNode:
    def __init__(self, name, connection_radius):
        self.action_server = actionlib.SimpleActionServer(name, ReactivePlannerAction, \
            execute_cb=self.action_callback, auto_start=False)
        
        self.action_feedback = ReactivePlannerFeedback()
        self.action_result = ReactivePlannerResult()
        self.action_server.start()

        self.curr_pos = None
        self.goal = None

        self.CONNECTION_RADIUS = connection_radius
        self.MAX_SENSOR_RANGE = 5

        self.laser_ranges = []    # [ang, dist]

        laser_sub = rospy.Subscriber('scan', LaserScan, callback=self.laser_callback)
        odom_sub = rospy.Subscriber('odom', Odometry, callback=self.odom_callback)

        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=100)

    def action_callback(self, req):
        self.goal = pointmsg_to_numpyarray(req.goal.pose.position)
        success = True
        start_time = rospy.Time.now()

        # error check
        if self.curr_pos is None:
            rospy.logerr('Robot position is unknown.')
            success = False

        if success:
            distance_to_goal = np.linalg.norm(self.curr_pos - self.goal)
            while distance_to_goal > 0.2:  # TODO: botar o 0.2 como parâmetro
                if self.action_server.is_preempt_requested():
                    vel = np.array([0, 0])
                    self.publish_velocity(vel)
                    rospy.loginfo('Action preempted.')
                    success = False
                    break

                vel = self.resultant_force()
                self.publish_velocity(vel)

                self.action_feedback.distance_to_goal = distance_to_goal
                self.action_server.publish_feedback(self.action_feedback)

            self.action_server.set_succeeded()
            self.action_result.time_duration = rospy.Time.now() - start_time
            rospy.loginfo('Action succeeded.')

        else:
            self.action_server.set_aborted()

    def laser_callback(self, msg):        
        aux = []
        min_ranges = []

        ranges = np.array(msg.ranges)
        it = np.nditer(ranges, flags=['c_index'])
        for r in it:
            if r >= self.MAX_SENSOR_RANGE - 0.5 or it.index == ranges.shape[0] - 1:
                if len(aux) > 0:
                    aux = np.array(aux)
                    min_dist_reading = aux[np.argmin(aux[:, 1])]
                    min_ranges.append(min_dist_reading)
                    # print(np.rad2deg(min_dist_reading[0]), min_dist_reading[1])
                    aux = []

                continue
            
            ang = it.index * msg.angle_increment + msg.angle_min
            aux.append([ang, r])
        
        self.laser_ranges = np.array(min_ranges)

    def odom_callback(self, msg):
        self.curr_pos = pointmsg_to_numpyarray(msg.pose.pose.position)
    
    def publish_velocity(self, vel):
        msg = Twist()
        msg.linear.x = vel[0]
        msg.linear.y = vel[1]
        
        self.vel_pub.publish(msg)

    def rep_gain(self, x, R, a=1, b=0):
        k = -a * (np.e ** -x)/(x * (x - R)) + b
        dir = np.sign(np.e ** -x * (x ** 2 - 2)/(x ** 2 - 2 * x) ** 2)   # dk / dx
        return k * dir

    def resultant_force(self):
        att = self.att_force()
        rep = self.rep_force(self.laser_ranges)
        
        res = att + rep
        print(rospy.get_name(), res)
        return res * 10

    def att_force(self, katt=1):
        curr_pos = self.curr_pos
        goal = self.goal

        if curr_pos is None:
            rospy.logerr('Robot position is unknown.')
            return

        if goal is None:
            rospy.logerr('Goal is unknown.')
            return
        
        return katt * (goal - curr_pos)

    def rep_force(self, laser_ranges, krep=2, max_dist=2):
        forces = np.array([0, 0])

        if len(self.laser_ranges) == 0: # no obstacles
            return np.array([0, 0])

        ang_it = np.nditer(laser_ranges[:, 0])
        d_it = np.nditer(laser_ranges[:, 1], flags=['c_index'])
        for ang, d in zip(ang_it, d_it):
            if d > max_dist:
                forces.append([0, 0])
            else:
                if self.is_robot(ang, d):
                    krep = self.rep_gain(d, self.CONNECTION_RADIUS)

                obs_pos = np.array([d * np.cos(ang), d * np.sin(ang)])
                force = krep * (d**2 * (1/d - 1/max_dist) / max_dist) * (obs_pos - self.curr_pos)
                forces += force
        
        return np.array(forces)

    # TODO checks if a given reading is of another robot or ordinary obstacle
    def is_robot(self, ang, d):
        return False


node_name = 'reactive_planner'
rospy.init_node(node_name)
rospy.loginfo(node_name + ' inicializado com sucesso.')
server = ReactiveNode(node_name, 2)
rospy.spin()