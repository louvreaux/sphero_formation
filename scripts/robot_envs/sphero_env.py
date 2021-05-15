import rospy
import message_filters as mf
import numpy as np

from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, Pose, PoseArray
from sphero_formation.msg import OdometryArray
from nav_msgs.msg import Odometry

import robot_stage_env
from util import Vector2

def get_agent_velocity(agent):
    """Return agent velocity as Vector2 instance."""
    vel = Vector2()
    vel.x = agent.twist.twist.linear.x
    vel.y = agent.twist.twist.linear.y
    return vel


def get_agent_position(agent):
    """Return agent position as Vector2 instance."""
    pos = Vector2()
    pos.x = agent.pose.pose.position.x
    pos.y = agent.pose.pose.position.y
    return pos


def get_obst_position(obst):
    """Return obstacle position as Vector2 instance."""
    pos = Vector2()
    pos.x = obst.position.x
    pos.y = obst.position.y
    return pos

class SpheroEnv(robot_stage_env.RobotStageEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self):

        rospy.init_node("sphero_dqlearn", anonymous=True, log_level=rospy.ERROR)

        rospy.logdebug("Start SpheroEnv INIT...")

        self.robot_name_space = "robot_5"

        # We launch the init function of the Parent Class robot_stage_env.RobotStageEnv
        super(SpheroEnv, self).__init__(robot_name_space=self.robot_name_space)
        # self.stage.unpauseSim()

        # Subscribers
        subs = [mf.Subscriber("robot_5/nearest", OdometryArray), mf.Subscriber("robot_5/avoid", PoseArray)]
        self.ts = mf.TimeSynchronizer(subs, 10)
        self.ts.registerCallback(self._callback)

        # Publishers
        self._cmd_vel_pub = rospy.Publisher('robot_5/cmd_vel', Twist, queue_size=1)

        temp_msg = Odometry()
        while not(abs(temp_msg.twist.twist.linear.y) > 0.05 or abs(temp_msg.twist.twist.linear.x) > 0.05):
            temp_msg = rospy.wait_for_message('/robot_0/odom', Odometry)

        # self.stage.pauseSim()
        
        rospy.logdebug("Finished SpheroEnv INIT...")
        

    def _callback(self, *data):
        my_agent = data[0].array[0]         # Odometry data for our agent is first in list
        nearest_agents = data[0].array[1:]  # Odometry data for neighbors follows
        obstacles = data[1].poses           # Store obstacles

        mean_position = Vector2()
        mean_velocity = Vector2()
        dist_array = np.array([])
        closest_agent_dist = np.inf

        # Init observation variables
        self.closest_obstacles = np.ones((10,2))                         # Relative position of 10 closest obstacles
        self.closest_neighbour = np.ones(2)                              # Relative position of closest neighbour
        self.flock_pose = np.ones(2)                                     # Relative position of flock
        self.flock_vel =  1.0                                            # Flock direction angle
        self.num_of_nearest_agents = len(nearest_agents)                 # Number of neighbours

        if nearest_agents:
            for agent in nearest_agents:
                agent_position = get_agent_position(agent)
                agent_velocity = get_agent_velocity(agent)
                mean_position += agent_position
                mean_velocity += agent_velocity

                if agent_position.norm() < closest_agent_dist:
                    self.closest_neighbour = np.array([agent_position.x, agent_position.y])
                    closest_agent_dist = agent_position.norm()

            # noise = np.random.normal(0.0, 0.03)     # Adding some noise for better generalization

            # Get closest neighbour observation
            # rospy.logerr("CLOSEST NEIGHBOUR X,Y: " + str(self.closest_neighbour))
            temp_dist = np.sqrt(self.closest_neighbour[0]**2 + self.closest_neighbour[1]**2)
            temp_angle = np.arccos(self.closest_neighbour[1] / temp_dist)
            if self.closest_neighbour[0] < 0:
                temp_angle = 2*np.pi - temp_angle
            self.closest_neighbour = np.array([temp_angle, temp_dist])                              # [angle, distance]

            # Get flock position observation
            self.flock_pose = np.array([mean_position.x, mean_position.y])/self.num_of_nearest_agents # + np.array([noise, noise])   # [X, Y]
            # rospy.logerr("FLOCK POSE X,Y: " + str(self.flock_pose))
            temp_dist = np.sqrt(self.flock_pose[0]**2 + self.flock_pose[1]**2)
            temp_angle = np.arccos(self.flock_pose[1] / temp_dist)
            if self.flock_pose[0] < 0:
                temp_angle = 2*np.pi - temp_angle
            self.flock_pose = np.array([temp_angle, temp_dist])                                     # [angle, distance]

            # Get direction angle of flock observation
            temp_var = np.array([mean_velocity.x, mean_velocity.y])/self.num_of_nearest_agents
            temp_var = Vector2(x=temp_var[0], y=temp_var[1])
            temp_var.normalize()
            self.flock_vel = np.arccos(temp_var.y) 
            if temp_var.x < 0:
                self.flock_vel = 2 * np.pi - self.flock_vel


        # Get 10 closest obstacles observation
        if obstacles:
            for obst in obstacles:
                temp_dist = Vector2(x = obst.position.x, y = obst.position.y)
                dist_array = np.append(dist_array, temp_dist.norm())

            counter = 0

            for x in np.argsort(dist_array):
                self.closest_obstacles[counter] = np.array([obstacles[x].position.x, obstacles[x].position.y])
                counter += 1
                if counter >= 10:
                    break

        # Get our agent direction angle
        temp_var = get_agent_velocity(my_agent)
        temp_var.normalize()

        temp = np.arccos(temp_var.y) # Angle between [0, 1] and normalized agent steering vector -> we only need y component
        if temp_var.x < 0:
           temp = 2*np.pi - temp # We need to take into account that angle between vectors is always the smallest one
        self.agent_steer = temp

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotStageEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_base(self, velocity):

        rospy.logdebug("Sphero Twist>>" + str(velocity))
        self._cmd_vel_pub.publish(velocity)

    def get_callback(self, curr_action, curr_flock_vel):
        # This is to make sure we observe the action our agent took
        # Flock velocity check is added because steer action can be the same as last time
        if curr_action != -1.0:
            while abs(self.agent_steer - curr_action) > 0.001 or curr_flock_vel == self.flock_vel:
                pass
        return self.agent_steer, self.closest_neighbour, self.flock_pose, self.flock_vel, self.closest_obstacles, self.num_of_nearest_agents