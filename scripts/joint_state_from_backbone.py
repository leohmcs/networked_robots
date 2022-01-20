#!/usr/bin/env python
#coding:utf-8

from os import link
from turtle import back, position
import rospy, tf2_ros
from tf import transformations

from moveit_msgs.msg import BoundingVolume, Constraints, MotionPlanRequest, PositionConstraint
from moveit_msgs.msg import MoveGroupActionGoal
from geometry_msgs.msg import PointStamped, Pose, Point, Vector3, Quaternion
from shape_msgs.msg import SolidPrimitive

import numpy as np

class JointStateFromBackbone:
    def __init__(self, net_radius=5):
        self.net_radius = net_radius
        self.links_names = ['robot4', 'robot3', 'robot2', 'robot1', 'robot0']

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.move_group_action_pub = rospy.Publisher('/move_group/goal', MoveGroupActionGoal, queue_size=100)

    def T(self, rot, trans):
        T = np.eye(4)
        for i in range(len(rot) - 1):
            r, p, y = rot[i + 1]
            t = np.c_[[0, 0, 5]]
            T_aux = np.vstack((np.hstack((self.R(r, p, y), t)), [0, 0, 0, 1]))
            T = np.matmul(T, T_aux)

        return T

    def invT(self, rot, trans):
        T = np.eye(4)
        for i in range(len(rot) - 1):
            r, p, y = rot[i + 1]
            t = [0, 0, 5]
            aux = np.hstack((self.invR(r, p, y), np.c_[-np.dot(self.invR(r, p, y), t)]))
            T_aux = np.vstack((aux, [0, 0, 0, 1]))

            T = np.matmul(T_aux, T)

        return T

    def R(self, r, p, y):
        aux = np.matmul(self.Rz(y), self.Ry(p))
        mat = np.matmul(aux, self.Rx(r))
        return mat

    def invR(self, r, p, y):
        # mat = np.transpose(self.R(r, p, y))
        aux = np.matmul(self.Rx(-r), self.Ry(-p))
        mat = np.matmul(aux, self.Rz(-y))
        return mat

    def Rz(self, th):
        mat = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        return mat

    def Ry(self, th):
        mat = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        return mat

    def Rx(self, th):
        mat = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
        return mat

    def transformation(self, target_frame, source_frame, time=0):
        if time == 0:
            time = rospy.Time()

        try:
            transform = self.tf_buffer.lookup_transform('robot2', 'base_link', time)

            trans = transform.transform.translation
            trans = np.array([trans.x, trans.y, trans.z])

            quat = transform.transform.rotation
            euler = transformations.euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))
            rot = self.invR(euler[0], euler[1], euler[2])


            return trans, rot
            
        except Exception as e:
            rospy.logerr('Error getting transformation from {} to {} at {}: {}'.format(source_frame, target_frame, time, e))
            return None, None

    def get_joint_pitch(self, p):
        '''p is the goal point in the link frame'''
        d = np.sqrt(p[0]**2 + p[1]**2)
        return np.sign(p[0])*(np.pi/2 - np.arccos(d/self.net_radius))

    def get_joint_yaw(self, p):
        '''p is the goal point in the link frame'''
        d = np.sqrt(p[0]**2 + p[1]**2)
        return -np.sign(p[1])*np.arcsin(p[1]/d)

    # TODO: as rotações dos frames ainda não funcionam direito
    def get_joints_angles(self, backbone):
        '''
        Returns the target joints angles so that the 2D projection of the manipulator in the xy-plane is equal to 
        a given backbone.
        Note: we don't calculate the angle of the joint that controls the end-effector orientation, because it is 
        not relevant in our case. one can simply assume that it's always 0.

        Input: backbone: the dict {robot name:robot position in 2D}. we assume the key of this dict is equal to 
        correspondent link name.
        Output: list of joint angles, from the base_link to the end-effector
        '''
        
        base_name = 'base_link'
        # joints_angles = {base_name:[0, 0]}
        joints_angles = [[0, 0, 0]]    # base_link is always [0, 0]
        links_positions = {base_name:[0, 0, 0]}
        previous_name = base_name
        for name in self.links_names:
            pos = None
            try:
                pos = np.array(backbone[name])
                pos = np.append(pos, 0)
            except KeyError:
                joints_angles.append([0, 0, 0])    # this robot is not part of the backbone
                links_positions[name] = [0, 0, 5]
                previous_name = name
                continue
            
            T = self.invT(joints_angles[1:], None)

            trans = np.array(links_positions[previous_name])
            posR = pos[:2] - trans[:2] # pos in robot R ("previous_name") frame
            
            q = np.array([pos[0], pos[1], np.sqrt(self.net_radius**2 - posR[0]**2 - posR[1]**2) + trans[2]])
            homo_q = np.append(q, 1)
            
            p = np.dot(T, homo_q)

            pitch = self.get_joint_pitch(p)
            yaw = self.get_joint_yaw(p)

            joints_angles.append([0, pitch, yaw])
            links_positions[name] = list(q)

            previous_name = name

        return joints_angles


    def constraints_msg(self, joints_constraints):
        '''
        Create a constraint message from required links positions. We assume all constraints have the same weight.
        TODO: allow different weights?

        Input: dict {link_name:required_link_position}
        Output: messge of type moveit_msgs/Constraints according to the required links positions
        '''

        constraints = []
        stamp = rospy.Time.now()

        for link in links_positions:
            position = links_positions[link]

            # create constraint region message
            volume = BoundingVolume()
            volume.primitives.append(SolidPrimitive(type=2, dimensions=[0.1]))
            volume.primitive_poses.append(Pose(Point(position[0], position[1], position[2]), Quaternion(0, 0, 0, 0)))

            msg = PositionConstraint()
            msg.header.frame_id = frame_id
            msg.header.stamp = stamp

            msg.link_name = link
            msg.target_point_offset = Vector3(offset, offset, offset)            
            msg.constraint_region = volume
            msg.weight = 1

            constraints.append(msg)

        return constraints

rospy.init_node('joint_state_from_backbone')


backbone = {'robot0': [6.81, -0.15], 'robot1': [5.35, -1.42], 'robot2': [3.94, -2.01], 'robot3': [2.53, -1.42]}

node = JointStateFromBackbone()
rate = rospy.Rate(10)
t = False
while not rospy.is_shutdown():
    if not t:
        angles = node.get_joints_angles(backbone)
        print(np.rad2deg(angles))
        t = True

    rate.sleep()