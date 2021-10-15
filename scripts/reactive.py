#!/usr/bin/env python
#coding: utf-8

import rospy
import actionlib
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, Twist
from interconnected_robots.msg import *
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan

from nav_msgs.srv import GetPlan

import numpy as np
import matplotlib.pyplot as plt

from utils.conversions import *
from utils.transformations import Rz


class ReactiveNode:
    def __init__(self, name, connection_radius):
        self.action_server = actionlib.SimpleActionServer(name, ReactivePlannerAction, \
            execute_cb=self.action_callback, auto_start=False)
        
        self.action_feedback = ReactivePlannerFeedback()
        self.action_result = ReactivePlannerResult()
        self.action_server.start()

        self.robot_pos = None
        self.robot_ori = None
        self.goal = None
        
        self.max_linear_speed = 0.3
        self.max_angular_speed = np.deg2rad(30)

        self.CONNECTION_RADIUS = connection_radius
        self.MAX_SENSOR_RANGE = 5

        self.dists_to_obstacles = []    # [ang, dist]

        laser_sub = rospy.Subscriber('scan', LaserScan, callback=self.laser_callback)
        odom_sub = rospy.Subscriber('odom', Odometry, callback=self.odom_callback)

        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=100)

    def action_callback(self, req):
        self.goal = pointmsg_to_numpyarray(req.goal.pose.position)
        success = True
        start_time = rospy.Time.now()

        # error check
        if self.robot_pos is None:
            rospy.logerr('{}\'s position is unknown.'.format(rospy.get_namespace()))
            success = False

        if self.robot_ori is None:
            rospy.logerr('{}\'s orientation is unknown.'.format(rospy.get_namespace()))
            success = False

        if success:
            distance_to_goal = np.linalg.norm(self.robot_pos - self.goal)
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
            if r >= self.MAX_SENSOR_RANGE or it.index == ranges.shape[0] - 1:
                if len(aux) > 0:                    
                    aux = np.array(aux)
                    min_dist_reading = aux[np.argmin(aux[:, 1])]
                    min_ranges.append(min_dist_reading)
                    aux = []

                continue
            
            ang = it.index * msg.angle_increment + msg.angle_min
            aux.append([ang, r])
        
        self.dists_to_obstacles = np.array(min_ranges)

    def odom_callback(self, msg):
        self.robot_pos = pointmsg_to_numpyarray(msg.pose.pose.position)
        quat = msg.pose.pose.orientation
        euler = tft.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.robot_ori = np.array(euler)
    
    def normalize_velocity(self, vel):
        return (vel / np.linalg.norm(vel)) * self.max_linear_speed

    def publish_velocity(self, vel):
        speed = np.linalg.norm(vel)
        if speed > self.max_linear_speed:
            vel = self.normalize_velocity(vel)

        msg = Twist()
        msg.linear.x = vel[0]
        msg.linear.y = vel[1]
        
        self.vel_pub.publish(msg)

    def rep_gain(self, x, R, a=2, b=2):
        k = -a * (np.e ** -x)/(x * (x - R)) + b
        dk = np.e**-x * (x**2 - R*x + 2*x - R)/((x**2 - R*x)**2)
        dir = -np.sign(dk)
        return k * dir

    def resultant_force(self):
        att = self.att_force()
        rep = self.rep_force(self.dists_to_obstacles, max_dist=self.MAX_SENSOR_RANGE)
        
        res = att + rep
        # self.animate(att, rep, res)
        return res

    def att_force(self, katt=.2):
        robot_pos = self.robot_pos
        goal = self.goal

        if robot_pos is None:
            rospy.logerr('{}\'s position is unknown.'.format(rospy.get_namespace()))
            return

        if goal is None:
            rospy.logerr('{}\'s goal is unknown.'.format(rospy.get_namespace()))
            return
        
        return katt * (goal - robot_pos)

    def rep_force(self, laser_ranges, krep=.5, max_dist=5):
        forces = np.array([0.0, 0.0])

        if len(self.dists_to_obstacles) == 0: # no obstacles
            return forces

        ang_it = np.nditer(laser_ranges[:, 0])
        d_it = np.nditer(laser_ranges[:, 1], flags=['c_index'])
        for ang, d in zip(ang_it, d_it):
            if d > max_dist:
                forces += [0.0, 0.0]
            else:
                if self.is_robot(ang, d):
                    krep = self.rep_gain(d, self.CONNECTION_RADIUS)
                obs_pos = np.array([d * np.cos(ang), d * np.sin(ang)]) + self.robot_pos
                force = - (krep / d**2) * (1/d - 1/max_dist) * (obs_pos - self.robot_pos)/max_dist
                forces += force
        
        return np.array(forces)

    # TODO checks if a given reading is of another robot or ordinary obstacle
    def is_robot(self, ang, d):
        return True

    def animate(self, att_force, rep_force, res_force):
        plt.cla()
        plt.quiver(self.robot_pos[0], self.robot_pos[1], att_force[0], att_force[1], color='r', label="Attraction")
        
        plt.quiver(self.robot_pos[0], self.robot_pos[1], rep_force[0], rep_force[1], color='g', label="Repulsion")
        
        plt.quiver(self.robot_pos[0], self.robot_pos[1], res_force[0], res_force[1], color='purple', label="Resultant Force")

        plt.plot(self.robot_pos[0], self.robot_pos[1], 'ro', label="Robot Position")
        plt.plot(self.goal[0], self.goal[1], 'go', label="Goal")
        
        plt.title(rospy.get_namespace().replace('/', ''))
        plt.legend()
        pad = 8
        width = (self.robot_pos[0] - pad, self.robot_pos[0] + pad)
        height = (self.robot_pos[1] - pad, self.robot_pos[1] + pad) 
        plt.axis((width[0], width[1], height[0], height[1]), 'equal')
        plt.draw()
        plt.pause(0.001)


node_name = 'reactive_planner'
rospy.init_node(node_name)
rospy.loginfo(node_name + ' inicializado com sucesso.')
server = ReactiveNode(node_name, 2)
rospy.spin()