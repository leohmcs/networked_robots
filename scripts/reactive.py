#!/usr/bin/env python
#coding: utf-8

import rospy
import actionlib
import tf.transformations as tft
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from interconnected_robots.msg import *
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
import tf

from nav_msgs.srv import GetPlan

import numpy as np
import matplotlib.pyplot as plt

from utils.conversions import *
from utils.transformations import Rz


class ReactiveNode:
    def __init__(self, name, connection_radius):
        # self.action_server = actionlib.SimpleActionServer(name, ReactivePlannerAction, \
        #     execute_cb=self.action_callback, auto_start=False)
        
        # self.action_feedback = ReactivePlannerFeedback()
        # self.action_result = ReactivePlannerResult()
        # self.action_server.start()

        self.robot_pos = None
        self.robot_ori = None
        self.goal = None
        
        self.max_linear_speed = 0.3
        self.min_linear_speed = 0.1
        self.max_angular_speed = np.deg2rad(30)

        self.CONNECTION_RADIUS = connection_radius
        self.MAX_SENSOR_RANGE = 5

        self.obstacles_positions = []    # [x, y]

        goal_sub = rospy.Subscriber('map/frontiers/nearest', PointStamped, callback=self.goal_callback)
        laser_sub = rospy.Subscriber('scan', LaserScan, callback=self.laser_callback)
        odom_sub = rospy.Subscriber('odom', Odometry, callback=self.odom_callback)

        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=100)

    def turn(self, ang_vel=None, angle=2*np.pi):
        if ang_vel is None:
            ang_vel = self.max_angular_speed

        dt = angle / ang_vel
        start_time = rospy.Time.now().to_sec()

        while rospy.Time.now().to_sec() - start_time < dt:
            vel = np.array([0.0, 0.0, ang_vel])
            self.publish_velocity(vel)

    def go(self):
        '''
        Go to goal position using potential fields
        '''
        if self.robot_pos is None:
            rospy.logerr('{}\'s position is unknown.'.format(rospy.get_namespace()))
            return

        if self.robot_ori is None:
            rospy.logerr('{}\'s orientation is unknown.'.format(rospy.get_namespace()))
            return

        if self.goal is None:
            rospy.loginfo('Waiting for goal')
            return

        distance_to_goal = np.linalg.norm(self.robot_pos - self.goal)
        while distance_to_goal > 0.7:  # TODO: botar como parâmetro
            print(distance_to_goal)
            vel = self.resultant_force()
            vel = np.append(vel, 0.0) # append angular velocity
            self.publish_velocity(vel)
            distance_to_goal = np.linalg.norm(self.robot_pos - self.goal)

    def goal_callback(self, msg):
        self.goal = np.array([msg.point.x, msg.point.y])

    def laser_callback(self, msg):
        robot_pos = self.robot_pos
        robot_yaw_rot = self.robot_ori[2]

        aux = []
        min_obstacles_pos = []     # closest point of each obstacle

        ranges = np.array(msg.ranges)
        it = np.nditer(ranges, flags=['c_index'])
        for r in it:
            if r >= self.MAX_SENSOR_RANGE or it.index == ranges.shape[0] - 1:
                if len(aux) > 0:                    
                    aux = np.array(aux)
                    min_dist_reading = aux[np.argmin(aux[:, 1])]
                    th, r = min_dist_reading[0], min_dist_reading[1]
                    reading_pos = [r * np.cos(th + robot_yaw_rot), r * np.sin(th + robot_yaw_rot)] + robot_pos
                    min_obstacles_pos.append(reading_pos)
                    aux = []

                continue
            
            ang = (it.index * msg.angle_increment) + msg.angle_min
            aux.append([ang, r])
        
        self.obstacles_positions = np.array(min_obstacles_pos)

    def odom_callback(self, msg):
        self.robot_pos = pointmsg_to_numpyarray(msg.pose.pose.position)
        quat = msg.pose.pose.orientation
        euler = tft.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.robot_ori = np.array(euler)

    def publish_velocity(self, vel):
        speed = np.linalg.norm(vel[:2])
        if speed > self.max_linear_speed and speed > 0:
            vel[:2] = vel[:2] / speed * self.max_linear_speed
        elif speed < self.min_linear_speed and speed > 0:
            vel[:2] = vel[:2] / speed * self.min_linear_speed

        msg = Twist()
        msg.linear.x = vel[0]
        msg.linear.y = vel[1]
        msg.angular.z = vel[2]
        
        self.vel_pub.publish(msg)

    def rep_gain(self, x, R, a=2, b=2):
        k = -a * (np.e ** -x)/(x * (x - R)) + b
        dk = np.e**-x * (x**2 - R*x + 2*x - R)/((x**2 - R*x)**2)
        dir = -np.sign(dk)
        return k * dir

    def resultant_force(self):
        att = self.att_force()
        rep = self.rep_force(self.obstacles_positions, max_dist=self.MAX_SENSOR_RANGE)
        
        res = att + rep
        self.animate(att, rep, res)
        return res

    def att_force(self, katt=.5):
        robot_pos = self.robot_pos
        goal = self.goal

        if robot_pos is None:
            rospy.logerr('{}\'s position is unknown.'.format(rospy.get_namespace()))
            return

        if goal is None:
            rospy.logerr('{}\'s goal is unknown.'.format(rospy.get_namespace()))
            return
        
        return katt * (goal - robot_pos)

    def rep_force(self, obs_pos, krep=.2, max_dist=2):
        forces = np.array([0.0, 0.0])

        if len(obs_pos) == 0: # no obstacles
            return forces

        # x = np.nditer(laser_ranges[:, 0])
        # y = np.nditer(laser_ranges[:, 1], flags=['c_index'])
        for p in obs_pos:
            d = np.linalg.norm(p - self.robot_pos)
            if d > max_dist:
                forces += [0.0, 0.0]
            else:
                if self.is_robot(p):
                    krep = self.rep_gain(d, self.CONNECTION_RADIUS)
                force = (krep / d**2) * ((1/d) - (1/max_dist)) * (self.robot_pos - p)/d
                forces += force
        
        return np.array(forces)

    # TODO checks if a given reading is of another robot or ordinary obstacle
    def is_robot(self, pos):
        return False

    def animate(self, att_force, rep_force, res_force):
        plt.cla()
        plt.quiver(self.robot_pos[0], self.robot_pos[1], att_force[0], att_force[1], color='g', label="Attraction")
        
        plt.quiver(self.robot_pos[0], self.robot_pos[1], rep_force[0], rep_force[1], color='r', label="Repulsion")
        
        plt.quiver(self.robot_pos[0], self.robot_pos[1], res_force[0], res_force[1], color='purple', label="Resultant Force")

        plt.plot(self.robot_pos[0], self.robot_pos[1], 'bo', label="Robot Position")
        plt.plot(self.goal[0], self.goal[1], 'go', label="Goal")
        obstacles = np.transpose(self.obstacles_positions)
        plt.scatter(obstacles[0], obstacles[1], color='r', label="Obstacles")

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
node = ReactiveNode(node_name, 2)
while not rospy.is_shutdown():
    node.turn()
    node.go()
    rospy.loginfo('[{}] Arrived'.format(rospy.get_namespace().replace('/', '')))
    rospy.sleep(0.1)