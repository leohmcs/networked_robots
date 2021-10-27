#!/usr/bin/env python
#coding: utf-8

import rospy
import actionlib
import tf.transformations as tft
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from interconnected_robots.msg import *
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
import tf2_ros

from nav_msgs.srv import GetPlan

import numpy as np
import matplotlib.pyplot as plt

from utils.conversions import *


class ReactiveNode:
    def __init__(self, namespace, connection_radius):
        self.namespace = namespace
        self.tf_buffer = tf2_ros.Buffer()
        tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.robot_pos = None
        self.robot_ori = None
        self.goal = None
        self.curr_vel = np.array([0.0, 0.0, 0.0])
        
        self.max_linear_speed = 0.3
        self.min_linear_speed = 0.1
        self.max_angular_speed = np.deg2rad(30)

        self.CONNECTION_RADIUS = connection_radius
        self.MAX_SENSOR_RANGE = 5

        self.obstacles_positions = []    # [x, y]

        goal_sub = rospy.Subscriber('map/frontiers/nearest', PointStamped, callback=self.goal_callback)
        laser_sub = rospy.Subscriber('scan', LaserScan, callback=self.laser_callback)
        odom_sub = rospy.Subscriber('odom', Odometry, callback=self.odom_callback)
        vel_sub = rospy.Subscriber('cmd_vel', Twist, callback=self.vel_callback)

        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=100)

    def goal_callback(self, msg):
        source_frame = msg.header.frame_id
        target_frame = self.namespace + '_tf/odom'
        try:
            tf = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time.now())
            self.goal = np.array([msg.point.x + tf.transform.translation.x, msg.point.y + tf.transform.translation.y])
        except Exception as e:
            rospy.logerr('Error transforming goal from frame {} to {}: {}'.format(source_frame, target_frame, str(e)))

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

    def vel_callback(self, msg):
        self.curr_vel = np.array([msg.linear.x, msg.linear.y, msg.angular.z])

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
    
    def ang(self, a, b):
        '''
        Returns the angle between vectors a and b
        '''
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return np.arccos(dot / (norm_a * norm_b))

    def turn(self, ang_vel=None, angle=2*np.pi):
        if ang_vel is None:
            ang_vel = self.max_angular_speed

        dt = angle / ang_vel
        start_time = rospy.Time.now().to_sec()

        while rospy.Time.now().to_sec() - start_time < dt:
            vel = np.array([0.0, 0.0, ang_vel])
            self.publish_velocity(vel)

    def go(self, min_goal_err=0.7):
        '''
        Go to goal position using potential fields
        Input 
        - min_goal_err: minimum distance to goal to consider the robot has arrived
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

        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal)
        while dist_to_goal > min_goal_err:
            l_vel = self.linear_velocity()  # [dx, dy]
            a_vel = self.angular_velocity() # [dth]
            q = np.append(l_vel, a_vel)       # [dx, dy, dth]
            self.publish_velocity(q)
            dist_to_goal = np.linalg.norm(self.robot_pos - self.goal)

    def linear_velocity(self):
        return self.resultant_force()

    def angular_velocity(self):
        target_ori = self.ang(self.goal - self.robot_pos, np.array([1, 0]))
        dth = target_ori - self.robot_ori[2]
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal)
        dt = self.max_linear_speed * dist_to_goal
        return dth / dt

    def rep_gain(self, x, R, a=2, b=2):
        k = -a * (np.e ** -x)/(x * (x - R)) + b
        dk = np.e**-x * (x**2 - R*x + 2*x - R)/((x**2 - R*x)**2)
        dir = -np.sign(dk)
        return k * dir

    def resultant_force(self):
        att = self.att_force()
        rep = self.rep_force(self.obstacles_positions, max_dist=self.MAX_SENSOR_RANGE)
        res = att + rep
        if np.deg2rad(-1) < np.abs(self.ang(att, rep)) - np.pi < np.deg2rad(1):
            res += self.noise_force(res)

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

    def noise_force(self, res):
        '''
        Adds an extra noise to help the robot escape local minima when the angle between repulsion 
        and atraction is around 180 degrees.

        The noise is in the direction the robot was moving and the module is the maximum robot speed.
        '''
        prev_lin_vel = self.curr_vel[:2]
        return self.max_linear_speed * (prev_lin_vel) / np.linalg.norm(prev_lin_vel)

    # TODO checks if a given reading is of another robot or obstacle
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
node = ReactiveNode(rospy.get_namespace().replace('/', ''), 2)
while not rospy.is_shutdown():
    node.turn()
    node.go()
    rospy.loginfo('[{}] Arrived'.format(rospy.get_namespace().replace('/', '')))
    rospy.sleep(0.1)