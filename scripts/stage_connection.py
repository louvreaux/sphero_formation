#!/usr/bin/env python

import rospy
import os

from std_srvs.srv import Empty
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

class StageConnection():

    def __init__(self):

        self.reset_simulation_proxy = rospy.ServiceProxy('/reset_positions', Empty)
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
        try:
            self.reset_simulation_proxy()
        except rospy.ServiceException as e:
            print ("/reset_position service call failed")