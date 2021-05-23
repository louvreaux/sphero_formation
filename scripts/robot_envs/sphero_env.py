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

def get_distance(array):
    
    return np.sqrt(array[0]**2 + array[1]**2)

def get_angle(array, dist):
    if dist == 0.0:
        angle = 0.0
    else:
        angle = np.arccos(array[0] / dist)
        if array[1] < 0.0:
            angle = 2 * np.pi - angle

    return angle


class SpheroEnv(robot_stage_env.RobotStageEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self):

        rospy.init_node("sphero_dqlearn", anonymous=True, log_level=rospy.ERROR)

        rospy.logdebug("Start SpheroEnv INIT...")

        self.robot_name_space = "robot_5"
        self.isCallback = False

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

        # Observation variables
        # self.closest_obstacles     =>  Relative position of 10 closest obstacles
        # self.closest_neighbour     =>  Relative position of closest neighbour
        # self.flock_pose            =>  Relative position of flock
        # self.flock_steer           =>  Flock direction angle
        # self.agent_steer           =>  Agent direction angle
        # self.steer_diff            =>  Flock - agent steer angle difference
        # self.num_of_nearest_agents =>  Number of neighbours

        # Get 10 closest obstacles observation
        self.closest_obstacles = np.ones((10, 2))
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
        
        self.num_of_nearest_agents = len(nearest_agents)
        
        # Get flock observations
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
            temp_dist = get_distance(self.closest_neighbour)
            temp_angle = get_angle(self.closest_neighbour, temp_dist)
            self.closest_neighbour = np.array([temp_angle, temp_dist]) # [angle, distance]

            # Get flock position observation
            temp_var = np.array([mean_position.x, mean_position.y])/self.num_of_nearest_agents # [X, Y]
            temp_dist = get_distance(temp_var)
            temp_angle = get_angle(temp_var, temp_dist)
            self.flock_pose = np.array([temp_angle, temp_dist])  # [angle, distance]

            # Get direction angle of flock observation
            temp_var_flock = np.array([mean_velocity.x, mean_velocity.y])/self.num_of_nearest_agents
            temp_dist = get_distance(temp_var_flock)
            temp_angle = get_angle(temp_var_flock, temp_dist)
            self.flock_steer = temp_angle
            
            # Get our agent direction angle
            temp_var_agent = get_agent_velocity(my_agent)
            temp_var_agent = np.array([temp_var_agent.x, temp_var_agent.y])
            temp_dist = get_distance(temp_var_agent)
            temp_angle = get_angle(temp_var_agent, temp_dist)
            self.agent_steer = temp_angle

            # Get steer angle difference
            self.steer_diff = np.arctan2(temp_var_flock[1], temp_var_flock[0]) - np.arctan2(temp_var_agent[1], temp_var_agent[0])
            if self.steer_diff > np.pi:
                self.steer_diff -= 2 * np.pi
            elif self.steer_diff <= -np.pi:
                self.steer_diff += 2 * np.pi

        self.isCallback = True

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

    def get_callback(self):
        # This is to make sure we observe the action our agent took
        # Flock velocity and pose check is added because steer action can be the same as step before
        while not self.isCallback:
            pass
        self.isCallback = False
        return self.steer_diff, self.closest_neighbour, self.flock_pose, self.flock_steer, self.closest_obstacles, self.num_of_nearest_agents