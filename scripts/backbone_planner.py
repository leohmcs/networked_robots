#!/usr/bin/env python
#coding: utf-8

import sys
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from moveit_msgs.msg import DisplayTrajectory, LinkPadding, LinkScale, PlanningScene
from moveit_msgs.srv import GetPositionFK
from networked_robots.msg import Backbone, BackboneTrajectory
from sensor_msgs.msg import JointState

import numpy as np
import kinematic_solver, octomap_generator


class BackbonePlanner(object):
    def __init__(self, num_planning_attempts=60, planning_time=45, planner_id="PRMstar", pipeline_id="ompl"):
        super(BackbonePlanner, self).__init__()

        # initialize moveit interfaces
        self.manipulator = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.group_name = 'backbone'
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.move_group.set_planning_pipeline_id(pipeline_id)
        self.move_group.set_planner_id(planner_id)
        self.move_group.set_num_planning_attempts(num_planning_attempts)
        self.move_group.set_planning_time(planning_time)
        # self.links_names = self.manipulator.get_link_names()
        # self.robots_names = ['robot4', 'robot3', 'robot2', 'robot1', 'robot0']
        # self.backbone = {}
        # self.kinematic_solver = kinematic_solver.KinematicSolver()
        # self.octomap_generator = octomap_generator.OctomapGenerator()

        # # self.groups_names = ['backbone1', 'backbone2', 'backbone3', 'backbone4', 'backbone5', 'backbone']
        # self.groups_names = self.manipulator.get_group_names()
        # self.move_groups = []
        # self.init_move_groups(self.groups_names)
        
        self.backbone_sub = rospy.Subscriber('backbone', Backbone, callback=self.backbone_callback)
        self.map_sub = rospy.Subscriber('map', OccupancyGrid, callback=self.map_callback)
        
        self.backbone_trajectory_pub = rospy.Publisher('backbone_trajectory', BackboneTrajectory, queue_size=10)
        self.display_trajectory_pub = rospy.Publisher('move_group/display_planned_path', DisplayTrajectory, queue_size=20)

        # forward kinematics service
        rospy.wait_for_service('compute_fk')
        self.fk_service = rospy.ServiceProxy('compute_fk', GetPositionFK)

    # TODO botar tempos diferentes para cada move group
    def init_move_groups(self, groups_names, pipeline_id='ompl', planner_id='BiTRRT', num_planning_attempts=50, planning_time=20):
        for group in groups_names:
            move_group = moveit_commander.MoveGroupCommander(group)
            move_group.set_planning_pipeline_id(pipeline_id)
            move_group.set_planner_id(planner_id)
            move_group.set_num_planning_attempts(num_planning_attempts)
            move_group.set_planning_time(planning_time)
            self.move_groups.append(move_group)

    def backbone_callback(self, msg):
        names = msg.names
        positions = msg.positions
        for i, name in enumerate(names):
            position = positions[i]
            self.backbone[name] = np.array([position.x, position.y])

    def map_callback(self, msg):
        self.update_planning_scene(msg)
    
    def update_planning_scene(self, occ_grid):
        ''' Updates the planning scene given a new Occupancy Grid. '''
        msg = PlanningScene()
        msg.name = 'mordor'
        msg.robot_state = self.move_groups[-1].get_current_state()
        msg.robot_model_name = 'backbone_manipulator'
        # TODO: self.scene.fixed_frame_transforms = precisa?
        msg.link_padding = [LinkPadding(link_name=name, padding=0.0) for name in self.links_names]
        msg.link_scale = [LinkScale(link_name=name, scale=1.0) for name in self.links_names]
        msg.world.octomap = self.octomap_generator.octomap_from_occupancygrid(occ_grid)
        msg.is_diff = True
        self.scene.apply_planning_scene(msg)

    def plan(self, backbone):
        joints_angles, num_robots = np.array(self.kinematic_solver.inverse_kinematics(backbone))
        # swap columns because we need yaw angles first (due to the form of our robot)
        joints_angles[:,[0, 1]] = joints_angles[:,[1, 0]]
        joints_angles = joints_angles.flatten()
        # joints_angles = joints_angles[-2*num_robots:]
        # joints_angles = np.append(joints_angles, 0.0)  # include end-effector joint goal
        # move_group = self.move_groups[num_robots - 1]   # -1 because python is 0-index
        move_group = self.move_groups[0]
        print('Using group {}.'.format(move_group.get_name()))
        move_group.set_joint_value_target(joints_angles)

        result = move_group.plan()
        if result[0] == True:
            plan = result[1]
            self.execute(plan, move_group)
            # self.format_plan_msg(plan)

    def execute(self, plan, move_group):
        move_group.execute(plan, wait=True)
        # cleanup after execution
        move_group.stop()
        move_group.clear_pose_targets()
        self.backbone = {}

    # TODO: usar serviço de cinemática inversa para obter posições dos links
    def get_robots_trajectories(self, plan):
        joint_state = JointState()
        joint_state.name = plan.joint_trajectory.joint_names
        points = plan.joint_trajectory.points
        paths = {robot:[] for robot in self.robots_names}   # each robot path is a list of PoseStamped
        times = []
        for point in points:
            joint_state.position = point.positions
            robots_positions = self.fk_service()
            for robot in robots_positions:
                pos = robots_positions[robot]
                pose = PoseStamped()
                pose.pose.position.x = pos[0]
                pose.pose.position.y = pos[1]
                pose.pose.orientation.w = 1.0
                paths[robot].append(pose)
            times.append(point.time_from_start)

        msg = BackboneTrajectory()
        msg.names = self.robots_names
        msg.paths = [Path(path) for path in paths]
        msg.time_from_start = times
        # publish robot's trajectories
        self.backbone_trajectory_pub.publish(msg)
        
    def display_trajectory(self, trajectory):
        display_trajectory_msg = DisplayTrajectory()
        display_trajectory_msg.trajectory_start = self.manipulator.get_current_state()
        display_trajectory_msg.trajectory.append(trajectory)
        self.display_trajectory_pub.publish(display_trajectory_msg)

    def print_info(self):
        print("============ Planning frame: %s" % self.move_group[-1].get_planning_frame())
        print("============ End effector link: %s" % self.move_group[-1].get_end_effector_link())
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
        planner.plan(planner.backbone)

    rate.sleep()