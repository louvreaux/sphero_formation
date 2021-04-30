#!/usr/bin/env python

import rospy
import random
import copy

from std_srvs.srv import Empty
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Pose2D

class StageConnection():

    def __init__(self):

        self.reset_simulation_proxy = rospy.ServiceProxy('/reset_positions', Empty)
        self.pause_simulation_proxy = rospy.ServiceProxy('/pause_sim', Empty)
        self.num_of_robots = rospy.get_param('num_of_robots')

        self.pub0 = rospy.Publisher('robot_0/cmd_pose', Pose2D, queue_size=1)
        self.pub1 = rospy.Publisher('robot_1/cmd_pose', Pose2D, queue_size=1)
        self.pub2 = rospy.Publisher('robot_2/cmd_pose', Pose2D, queue_size=1)
        self.pub3 = rospy.Publisher('robot_3/cmd_pose', Pose2D, queue_size=1)
        self.pub4 = rospy.Publisher('robot_4/cmd_pose', Pose2D, queue_size=1)
        self.pub5 = rospy.Publisher('robot_5/cmd_pose', Pose2D, queue_size=1)

        #self.pose_list = [Pose2D(x=0.3, y=0.2), Pose2D(x=-0.3, y=0.2), Pose2D(x=0.3, y=-0.2), Pose2D(x=-0.3, y=-0.2), Pose2D(x=0.0, y=0.0), Pose2D(x=0.0, y=0.4)]
        self.pose_list = [Pose2D(x=-3.7, y=0.2), Pose2D(x=-4.3, y=0.2), Pose2D(x=-3.7, y=-0.2), Pose2D(x=-4.3, y=-0.2), Pose2D(x=-4.0, y=0.0), Pose2D(x=-4.0, y=0.4)]
        self.rand_x = -4.0
        self.rand_y = 0.0

        self.counter = 0
        #self.pauseSim()

    def pauseSim(self):
        rospy.logdebug("PAUSING...")
        rospy.wait_for_service('/pause_sim')
        try:
           self.pause_simulation_proxy()
        except rospy.ServiceException as e:
           print ("/pause_sim service call failed")

    def unpauseSim(self):
        rospy.logdebug("UNPAUSING...")
        rospy.wait_for_service('/pause_sim')
        try:
           self.pause_simulation_proxy()
        except rospy.ServiceException as e:
           print ("/pause_sim service call failed")

    def resetSim(self):
        rospy.logdebug("RESETING POSITIONS...")
        if self.counter == 20:
            random.shuffle(self.pose_list)

            #self.rand_x = (random.random() * 2.0 - 1.0) * 4.0
            #self.rand_y = (random.random() * 2.0 - 1.0) * 4.0

            self.counter = 0
        else:
            self.counter += 1

        temp_pose_list = copy.deepcopy(self.pose_list)

        # for i in range(self.num_of_robots):

        #     temp_pose_list[i].x = self.pose_list[i].x + self.rand_x
        #     temp_pose_list[i].y = self.pose_list[i].y + self.rand_y

        self.pub0.publish(temp_pose_list[0])
        self.pub1.publish(temp_pose_list[1])
        self.pub2.publish(temp_pose_list[2])
        self.pub3.publish(temp_pose_list[3])
        self.pub4.publish(temp_pose_list[4])
        self.pub5.publish(temp_pose_list[5])

        # rospy.wait_for_service('/reset_positions')
        # try:
        #     self.reset_simulation_proxy()
        # except rospy.ServiceException as e:
        #     print ("/reset_position service call failed")