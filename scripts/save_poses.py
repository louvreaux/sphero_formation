#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
import message_filters as mf
import numpy as np
import threading


class SavePose():

    def __init__(self):

        self.map_msg = rospy.wait_for_message("map", OccupancyGrid).data
        np.savetxt("/home/lovro/sphero_ws/src/sphero_formation/training_results/DDPG/map.csv", self.map_msg, delimiter=",")
        subs = [mf.Subscriber("robot_0/odom", Odometry), mf.Subscriber("robot_1/odom", Odometry), 
                mf.Subscriber("robot_2/odom", Odometry), mf.Subscriber("robot_3/odom", Odometry),
                mf.Subscriber("robot_4/odom", Odometry), mf.Subscriber("robot_5/odom", Odometry)]
        
        self.ts = mf.TimeSynchronizer(subs, 10)
        self.ts.registerCallback(self.cb_update)
        self.lock = threading.Lock()

        self.pose_list = np.array([])

    def cb_update(self, *data):
        temp_lst = np.array([])
        for d in data:
            temp_lst = np.append(temp_lst, [d.pose.pose.position.x, d.pose.pose.position.y])
        temp_lst = np.reshape(temp_lst, (-1,2))
        
        self.pose_list = np.hstack((self.pose_list, temp_lst)) if self.pose_list.size else temp_lst
    
    def get_list(self):
        return self.pose_list
    
    def erase_list(self):
        self.pose_list = np.array([])
    
    def get_map(self):
        return self.map_msg