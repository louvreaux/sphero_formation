#!/usr/bin/env python
from __future__ import print_function
import rospy
import math
from copy import deepcopy
import message_filters as mf
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import Odometry, OccupancyGrid
from sphero_formation.msg import OdometryArray


def get_distance(a, b):
    """Return Euclidean distance between points a and b."""
    return math.sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2))


class NearestSearch():
    def map_callback(self, data):
        """Save map metadata in class variables."""
        self.map = data.data
        self.map_width = data.info.width
        self.map_height = data.info.height
        self.map_resolution = data.info.resolution
        self.map_origin = data.info.origin.position

    def pos_to_index(self, x_real, y_real):
        """Return list indexes for given real position coordinates."""
        x_ind = (x_real + self.map_origin.x) / self.map_resolution
        y_ind = (y_real - self.map_origin.y) / self.map_resolution
        return int(x_ind), int(y_ind)

    def index_to_pos(self, x_ind, y_ind):
        """Return real position coordinates for list indexes."""
        x_real = x_ind * self.map_resolution - self.map_origin.x
        y_real = y_ind * self.map_resolution + self.map_origin.y
        return x_real, y_real

    def robot_callback(self, *data):
        """Find and publish positions of nearest agents and obstacles."""
        search_radius = rospy.get_param('/dyn_reconf/search_radius')
        avoid_radius = rospy.get_param('/dyn_reconf/avoid_radius')
        r = int(avoid_radius / self.map_resolution)

        for agent in data:
            # Get current time and observed agent's name
            time = rospy.Time.now()
            key = agent.header.frame_id.split('/')[1]

            # Get agent's position and initialize relative positions message
            agent_position = agent.pose.pose.position
            nearest_agents = OdometryArray()
            nearest_agents.header.stamp = time

            # For all agents calculate the distance from the observed agent
            # If the distance is within specified, add the agent to the
            # publishing message
            for agent2 in data:
                friend_position = agent2.pose.pose.position
                distance = get_distance(agent_position, friend_position)
                if distance > 0 and distance <= search_radius:
                    # Calculate relative position
                    rel_pos = deepcopy(agent2)
                    rel_pos.pose.pose.position.x = friend_position.x - agent_position.x
                    rel_pos.pose.pose.position.y = friend_position.y - agent_position.y
                    nearest_agents.array.append(rel_pos)

            # Publish the relative positions message
            self.nearest[key].publish(nearest_agents)

            # Initialize the wall and obstacle positions message
            avoids = PoseArray()
            avoids.header.stamp = time

            # Positions of walls and obstacles are represented as value 100 in a
            # list `self.map`. First, find the position of the observed agent in
            # this list in form of an index pair. Then search the list in
            # in specified search radius and return actual positions of walls
            # other obstcles.
            x0, y0 = self.pos_to_index(agent_position.x, agent_position.y)
            x_range = range(max(0, x0 - r / 2), min(self.map_height, x0 + r / 2))
            y_range = range(max(0, y0 - r / 2), min(self.map_width, y0 + r / 2))
            for i in x_range:
                for j in y_range:
                    print(self.map[i * self.map_width + j], end=' ')
                    if self.map[i * self.map_width + j] == 100:
                        x, y = self.index_to_pos(i, j)
                        obst = Pose()
                        obst.position.x = x - agent_position.x
                        obst.position.y = y - agent_position.y
                        avoids.poses.append(obst)
                print()
            print()

            # Publish the wall and obstacle positions message
            self.avoid[key].publish(avoids)

    def __init__(self):
        # TODO: Naci bolji nacin za ovo
        self.num_agents = rospy.get_param("/search/num_of_robots")

        # Create a publisher for commands
        pub_keys = ['robot_{}'.format(i) for i in range(self.num_agents)]

        # Publisher for locations of nearest agents
        self.nearest = dict.fromkeys(pub_keys)
        for key in self.nearest.keys():
            self.nearest[key] = rospy.Publisher('/' + key + '/nearest', OdometryArray, queue_size=1)

        # Publisher for locations of walls and obstacles
        self.avoid = dict.fromkeys(pub_keys)
        for key in self.avoid.keys():
            self.avoid[key] = rospy.Publisher('/' + key + '/avoid', PoseArray, queue_size=1)

        # Create a subscriber
        subs = [mf.Subscriber("/robot_{}/odom".format(i), Odometry) for i in range(self.num_agents)]
        self.ts = mf.TimeSynchronizer(subs, 10)
        self.ts.registerCallback(self.robot_callback)

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)

        # Main while loop.
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('NearestSearch')

    # Go to class functions that do all the heavy lifting.
    # Do error checking.
    try:
        ns = NearestSearch()
    except rospy.ROSInterruptException:
        pass
