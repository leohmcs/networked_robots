#!/usr/bin/env python
#coding: utf-8

import sys
import rospy
import moveit_commander
from nav_msgs.msg import OccupancyGrid
from moveit_msgs.msg import DisplayTrajectory, LinkPadding, LinkScale, PlanningScene
from networked_robots.msg import Backbone

import pyvisgraph as vg
import numpy as np
import goals_planner, joint_state_from_backbone, octomap_generator


class BackbonePlanner(object):
    def __init__(self, num_planning_attempts=40, planning_time=10, planner_id="RRTstar", pipeline_id="ompl"):
        super(BackbonePlanner, self).__init__()
        self.links_names = ['base_link', 'base', 'network_base_to_robot4', 'robot4', 'network_robot4_to_robot3', 'robot3', 'network_robot3_to_robot2', 'robot2', 'network_robot2_to_robot1', 'robot1', 'network_robot1_to_robot0', 'robot0']   # TODO
        self.backbone = {}
        self.joint_state_from_backbone = joint_state_from_backbone.JointStateFromBackbone()
        self.octomap_generator = octomap_generator.OctomapGenerator()

        self.manipulator = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.group_name = 'backbone'
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.move_group.set_num_planning_attempts(num_planning_attempts)
        self.move_group.set_planning_time(planning_time)
        self.move_group.set_planner_id(planner_id)
        self.move_group.set_planning_pipeline_id(pipeline_id)
        
        # listen required backbone configuration
        self.backbone_sub = rospy.Subscriber('backbone', Backbone, callback=self.backbone_callback)

        self.map_sub = rospy.Subscriber('map', OccupancyGrid, callback=self.map_callback)
        
        # display planned trajectory in RViz
        self.display_trajectory_pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=20)
        
        # test
        example_path = [vg.Point(-3.94, -0.01), vg.Point(-1.41, -1.41), vg.Point(0.00, -2.00), vg.Point(1.41, -1.41), vg.Point(2.87, -0.14)]
        self.goals_planner = goals_planner.GoalsPlanner(example_path, 5, None)
        self.base_position = [example_path[0].x, example_path[0].y]

    def backbone_callback(self, msg):
        names = msg.names
        positions = msg.positions

        for i, name in enumerate(names):
            position = positions[i]
            self.backbone[name] = np.array([position.x, position.y])

    def map_callback(self, msg):
        self.update_planning_scene(msg)

    def plan_motion(self, backbone):
        joints_angles = np.array(self.joint_state_from_backbone.get_joints_angles(backbone))
        # swap columns because we need yaw angles first (due to the form of our robot)
        joints_angles[:,[0, 1]] = joints_angles[:,[1, 0]]
        joints_angles = joints_angles.flatten()
        # joints_angles = np.append(joints_angles, 0.0)  # include end-effector joint goal
        self.move_group.set_joint_value_target(joints_angles)

        self.execute()

    def execute(self):
        self.move_group.go(wait=True)

        # cleanup after execution
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        self.backbone = {}

    def update_planning_scene(self, occ_grid):
        ''' Updates the planning scene given a new Occupancy Grid. '''
        msg = PlanningScene()
        msg.name = 'mordor'
        msg.robot_state = self.move_group.get_current_state()
        msg.robot_model_name = 'backbone_manipulator'
        # TODO: self.scene.fixed_frame_transforms = precisa?
        msg.link_padding = [LinkPadding(link_name=name, padding=0.0) for name in self.links_names]
        msg.link_scale = [LinkScale(link_name=name, scale=1.0) for name in self.links_names]
        msg.world.octomap = self.octomap_generator.octomap_from_occupancygrid(occ_grid)
        msg.is_diff = True

        self.scene.apply_planning_scene(msg)
        
    def display_trajectory(self, trajectory):
        display_trajectory_msg = DisplayTrajectory()
        display_trajectory_msg.trajectory_start = self.manipulator.get_current_state()
        display_trajectory_msg.trajectory.append(trajectory)
        self.display_trajectory_pub.publish(display_trajectory_msg)

    def print_info(self):
        print("============ Planning frame: %s" % self.move_group.get_planning_frame())
        print("============ End effector link: %s" % self.move_group.get_end_effector_link())
        print("============ Available Planning Groups:", self.manipulator.get_group_names())
        print("============ Printing robot state")
        print(self.robot.get_current_state())
        print("")


moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('backbone_planner', anonymous=True)

planner = BackbonePlanner()

rate = rospy.Rate(0.2)
while not rospy.is_shutdown():
    if len(planner.backbone) > 0:
        planner.plan_motion(planner.backbone)

    rate.sleep()