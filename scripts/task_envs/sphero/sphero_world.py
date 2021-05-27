import rospy
import numpy as np

from gym import spaces
from gym.envs.registration import register
from geometry_msgs.msg import Twist
from math import cos, sin
from robot_envs import sphero_env

timestep_limit_per_episode = 2500 # Can be any Value

register(
        id='SpheroWorld-v0',
        entry_point='task_envs.sphero.sphero_world:SpheroWorldEnv',
        max_episode_steps=timestep_limit_per_episode,
    )

class SpheroWorldEnv(sphero_env.SpheroEnv):
    def __init__(self):

        # Load needed parameters

        # Discretization steps
        self.quant_step = rospy.get_param('quantization_step')
        self.angle_quant_step = rospy.get_param('angle_quantization_step')

        # Observation limits
        self.vel_max = rospy.get_param('max_velocity')
        self.rel_pose_max = rospy.get_param('max_relative_pose')

        # Radiuses
        self.too_close = rospy.get_param('too_close')


        # Define action and observation space
        self.action_space = spaces.Discrete(int(360.0/self.angle_quant_step))

        self.observation_space = spaces.Box(low=np.array([-1*np.pi, 0.0, 0.0, 0.0, 0.0]), 
                                            high=np.array([np.pi, 2*np.pi, self.rel_pose_max, 2*np.pi, self.rel_pose_max]), 
                                            dtype=np.float64)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)
        
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.cumulated_steps = 0.0
        self.crash = False

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(SpheroWorldEnv, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_base(Twist())

        return True


    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False


    def _set_action(self, action):
        """
        This set action will set speed of the Sphero robot
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        theta = action * self.angle_quant_step * (np.pi/180.0)
        new_vel = np.array([-sin(-theta), cos(-theta)]) * self.vel_max # efektivno rotiramo vektor [0, 1] u desno (jer tako gledamo i kut gibanja jata)
        
        send_velocity = Twist()
        send_velocity.linear.x = new_vel[0]
        send_velocity.linear.y = new_vel[1]
        self.move_base(send_velocity)
        
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        steer_diff, closest_neighbour, flock_pose, flock_steer, closest_obst, num_of_neighbours = self.get_callback()
        
        discretized_observations = self.discretize_observation(steer_diff, closest_neighbour, flock_pose, flock_steer, closest_obst, num_of_neighbours)

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")
        rospy.logwarn(discretized_observations)

        return discretized_observations
        

    def _is_done(self, observations):
        
        if self._episode_done:
            if self.crash:
                rospy.logerr("Sphero went too close to his neighbour!")
                self.crash = False
            else:
                rospy.logerr("Sphero diverged from the flock!")

        return self._episode_done

    def _compute_reward(self, observations, done):

        reward = 0.0

        if not done:

            # Punish being too close to neighbour
            reward += (5.0 / (1.0 + np.exp(-50.0 * (observations[1] - 0.2))) - 5.0)
            # temp = reward
            # rospy.logerr("CLOSEST NEIGHBOUR REWARD: " + str(reward))

            # Reward being close to flock
            reward += (5.0 - 5.0 / (1.0 + np.exp(-20.0 * (observations[4] - 0.35))))
            # rospy.logerr("FLOCK REWARD: " + str(reward - temp))
            # temp = reward

            # Reward going the similar way as flock
            reward += (5.0 - 5.0 / (1.0 + np.exp(-4.0 * (abs(observations[0] - np.pi/2.0)))))
            # rospy.logerr("STEER REWARD: " + str(reward - temp))
        else:
             reward -= 10

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward


    # Internal TaskEnv Methods
    
    def discretize_observation(self, steer_diff, closest_neighbour, flock_pose, flock_steer, closest_obst, num_of_neighbours):
        """

        """
        self._episode_done = False
        if  num_of_neighbours != 0:

            # KVANTIZIRAJ SMJER AGENTA
            steer_diff = self.observation_quantization(steer_diff, -1*np.pi, np.pi, self.angle_quant_step * np.pi / 180.0)
            if abs(steer_diff) <= 0.001:
                steer_diff = 0.0

            # KVANTIZIRAJ POZICIJU NAJBLIZEG AGENTA I PREPREKA
            closest_neighbour[0] = self.observation_quantization(closest_neighbour[0], 0.0, 2*np.pi, self.angle_quant_step * np.pi / 180.0)
            closest_neighbour[1] = self.observation_quantization(closest_neighbour[1], 0.0, self.rel_pose_max, self.quant_step)

            if abs(closest_neighbour[0]) <= 0.001:
                closest_neighbour[0] = 0.0

            if closest_neighbour[1] <= self.too_close:
                self.crash = True
                self._episode_done = True

            # KVANTIZIRAJ SREDNJU POZICIJU SUSJEDA
            flock_pose[0] = self.observation_quantization(flock_pose[0], 0.0, 2*np.pi, self.angle_quant_step * np.pi / 180.0)
            flock_pose[1] = self.observation_quantization(flock_pose[1], 0.0, self.rel_pose_max, self.quant_step)

            if abs(flock_pose[0]) <= 0.001:
                flock_pose[0] = 0.0

            discretized_obs = np.array([steer_diff, closest_neighbour[0], closest_neighbour[1], flock_pose[0], flock_pose[1]])
        else:
            self._episode_done = True
            discretized_obs = self.last_obs

        self.last_obs = discretized_obs

        return discretized_obs

    def observation_quantization(self, value, min_value, max_value, qstep):
        if np.greater(value, max_value):
            value = max_value
        elif np.less(value, min_value):
            value = min_value

        return  qstep * np.round(value/qstep)