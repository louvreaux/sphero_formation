import rospy
import message_filters as mf
import numpy as np

from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, Pose, PoseArray
from sphero_formation.msg import OdometryArray

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

        rospy.logdebug("Start SpheroEnv INIT...")

        self.robot_name_space = "robot_5"

        # We launch the init function of the Parent Class robot_stage_env.RobotStageEnv
        super(SpheroEnv, self).__init__(robot_name_space=self.robot_name_space)
        self.stage.unpauseSim()

        # Subscribers
        subs = [mf.Subscriber("robot_5/nearest", OdometryArray), mf.Subscriber("robot_5/avoid", PoseArray)]
        self.ts = mf.TimeSynchronizer(subs, 10)
        self.ts.registerCallback(self._callback)

        # Publishers
        self._cmd_vel_pub = rospy.Publisher('robot_5/cmd_vel', Twist, queue_size=1)

        # rospy.wait_for_message('robot_5/nearest', OdometryArray)
        self.stage.pauseSim()
        
        rospy.logdebug("Finished SpheroEnv INIT...")
        

    def _callback(self, *data):
        my_agent = data[0].array[0]         # odometry data for this agent is first in list
        nearest_agents = data[0].array[1:]  # odometry data for neighbors follows
        obstacles = data[1].poses           # store obstacles

        mean_position = Vector2()
        mean_velocity = Vector2()
        dist_array = np.array([])
        closest_agent_dist = np.inf

        temp_var = get_agent_velocity(my_agent)
        temp_var.normalize()
        self.agent_steer = np.arccos(temp_var.y) # kut izmedju [0, 1] i jedinicnog smjera gibanja => potrebna samo y komponenta
        if temp_var.x < 0:
           self.agent_steer += np.pi                                     # SMJER GIBANJA NASEG AGENTA

        self.closest_obstacles = np.ones((10,2))                         # RELATIVNA POZICIJA NAJBLIZIH PREPREKA
        self.closest_agent_pose = np.ones(2)                             # RELATIVNA POZICIJA NAJBLIZEG SUSJEDA
        self.direction = np.ones(2)                                      # RELATIVNA SREDNJA POZICIJA SUSJEDA
        self.steer =  10.0                                               # RELATIVNA SREDNJA BRZINA SUSJEDA
        self.num_of_nearest_agents = len(nearest_agents)

        # IZRACUN PROSJECNE POZICIJE I BRZINE NAJBLIZIH AGENATA
        if nearest_agents:
            for agent in nearest_agents:
                agent_position = get_agent_position(agent)
                agent_velocity = get_agent_velocity(agent)
                mean_position += agent_position
                mean_velocity += agent_velocity

                if agent_position.norm() < closest_agent_dist:
                    self.closest_agent_pose = np.array([agent_position.x, agent_position.y])
                    closest_agent_dist = agent_position.norm()

            noise = np.random.normal(0.0, 0.03)
            self.closest_agent_pose = self.closest_agent_pose + np.array([noise, noise])
            temp_var = mean_position/self.num_of_nearest_agents
            self.direction = np.array([temp_var.x, temp_var.y]) + np.array([noise, noise])

            temp_var = mean_velocity/self.num_of_nearest_agents
            temp_var.normalize()
            self.steer = np.arccos(temp_var.y) # kut izmedju [0, 1] i jedinicnog smjera gibanja => potrebna samo y komponenta
            if temp_var.x < 0:
                self.steer += np.pi


        # IZRACUN NAJBLIZIH PREPREKA
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

        # TU JE BILA FUNKCIJA KOJA CEKA DA ROBOT PROMIJENI POZICIJU TEMELJEM AKCIJE
    def get_callback(self):

        return self.agent_steer, self.closest_agent_pose, self.direction, self.steer, self.closest_obstacles, self.num_of_nearest_agents