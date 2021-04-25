#!/usr/bin/env python

import rospy
import os
import random

from std_srvs.srv import Empty
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Pose2D

class StageConnection():

    def __init__(self):

        self.reset_simulation_proxy = rospy.ServiceProxy('/reset_positions', Empty)
        self.num_of_robots = rospy.get_param('num_of_robots')

        self.pub0 = rospy.Publisher('robot_0/cmd_pose', Pose2D, queue_size=1); self.r0_init_pose = Pose2D(x=0.3, y=0.2)
        self.pub1 = rospy.Publisher('robot_1/cmd_pose', Pose2D, queue_size=1); self.r1_init_pose = Pose2D(x=-0.3, y=0.2)
        self.pub2 = rospy.Publisher('robot_2/cmd_pose', Pose2D, queue_size=1); self.r2_init_pose = Pose2D(x=0.3, y=-0.2)
        self.pub3 = rospy.Publisher('robot_3/cmd_pose', Pose2D, queue_size=1); self.r3_init_pose = Pose2D(x=-0.3, y=-0.2)
        self.pub4 = rospy.Publisher('robot_4/cmd_pose', Pose2D, queue_size=1); self.r4_init_pose = Pose2D(x=0.0, y=0.0)
        self.pub5 = rospy.Publisher('robot_5/cmd_pose', Pose2D, queue_size=1); self.r5_init_pose = Pose2D(x=0.0, y=0.4)
        # self.pauseSim()

    def pauseSim(self):
        rospy.logdebug("PAUSING...")
        os.system('xdotool windowactivate $(xdotool search --name Stage) && xdotool key p && xdotool windowactivate $(xdotool getactivewindow)')

    def unpauseSim(self):
        rospy.logdebug("UNPAUSING...")
        os.system('xdotool windowactivate $(xdotool search --name Stage) && xdotool key p && xdotool windowactivate $(xdotool getactivewindow)')

    def resetSim(self):
        rospy.logdebug("RESETING POSITIONS...")
        rospy.wait_for_service('/reset_positions')
        rand_x = (random.random() * 2.0 - 1.0) * 4.0
        rand_y = (random.random() * 2.0 - 1.0) * 4.0

        temp_pose = Pose2D()
        temp_pose.x = self.r0_init_pose.x + rand_x; temp_pose.y = self.r0_init_pose.y + rand_y
        self.pub0.publish(temp_pose)

        temp_pose = Pose2D()
        temp_pose.x = self.r1_init_pose.x + rand_x; temp_pose.y = self.r1_init_pose.y + rand_y
        self.pub1.publish(temp_pose)

        temp_pose = Pose2D()
        temp_pose.x = self.r2_init_pose.x + rand_x; temp_pose.y = self.r2_init_pose.y + rand_y
        self.pub2.publish(temp_pose)

        temp_pose = Pose2D()
        temp_pose.x = self.r3_init_pose.x + rand_x; temp_pose.y = self.r3_init_pose.y + rand_y
        self.pub3.publish(temp_pose)

        temp_pose = Pose2D()
        temp_pose.x = self.r4_init_pose.x + rand_x; temp_pose.y = self.r4_init_pose.y + rand_y
        self.pub4.publish(temp_pose)

        temp_pose = Pose2D()
        temp_pose.x = self.r5_init_pose.x + rand_x; temp_pose.y = self.r5_init_pose.y + rand_y
        self.pub5.publish(temp_pose)

        #try:
        #    self.reset_simulation_proxy()
        #except rospy.ServiceException as e:
        #    print ("/reset_position service call failed")