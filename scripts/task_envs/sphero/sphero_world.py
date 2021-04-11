import rospy
import numpy as np

from gym import spaces
from gym.envs.registration import register
from geometry_msgs.msg import Vector3, Twist
from math import sqrt, cos, sin
from robot_envs import sphero_env

timestep_limit_per_episode = 1000 # Can be any Value

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
        self.pose_max = rospy.get_param('max_pose')
        self.rel_pose_max = rospy.get_param('max_relative_pose')
        self.rel_obst_pose_max = rospy.get_param('max_relative_pose_to_obstacle')

        # Radiuses
        self.too_close = rospy.get_param('too_close')               # agent shoudn't be inside of this radius (it means that it has crashed)
        self.crowd_radius_q = rospy.get_param('crowd_radius_q')     # agent shouldn't be inside of this radius (too close to neighbours)
        self.avoid_radius_q = rospy.get_param('avoid_radius_q')     # agent shouldn't be inside of this radius (too close to obstacles)
        self.close_radius = rospy.get_param('close_radius')         # agent should be inside of this radius

        self.end_episode_points = rospy.get_param('end_episode_points')


        # Define action and observation space
        self.action_space = spaces.Discrete(int(360.0/self.angle_quant_step))

        self.observation_space = spaces.Tuple((
                    spaces.Box(low = 0.0, high = 2*np.pi, shape = (1,1),  dtype = np.float64),                                          # smjer
                    spaces.Box(low = -self.rel_pose_max, high = self.rel_pose_max, shape = (1,2), dtype = np.float64),                  # najblizi susjed
                    spaces.Box(low = -self.rel_pose_max, high = self.rel_pose_max, shape = (1,2), dtype = np.float64),                  # pozicija grupiranja
                    spaces.Box(low = 0.0, high = 2*np.pi, shape = (1,1), dtype = np.float64)))                                          # smjer jata
                    #spaces.Box(low = -self.rel_obst_pose_max, high = self.rel_obst_pose_max, shape = (10, 2), dtype = np.float64)))    # pozicije prepreka

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)
        
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.cumulated_steps = 0.0

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
        agent_steer, closest_agent_pose, direction, steer, closest_obst = self.get_callback()
        
        discretized_observations = self.discretize_observation(agent_steer, closest_agent_pose, direction, steer)#, closest_obst)
        self.disc_speed = discretized_observations[1]

        rospy.logdebug("Observations==>"+str(discretized_observations))
        rospy.logdebug("END Get Observation ==>")
        rospy.logwarn(discretized_observations)

        return discretized_observations
        

    def _is_done(self, observations):
        
        if self._episode_done:
            rospy.logerr("Sphero has diverged from flock or has hit something!")

        return self._episode_done

    def _compute_reward(self, observations, done):

        reward = 0.0

        if not done:

            # OVO JE DA NE BUDE BLIZU NAJBLIZEM AGENTU
            dist_to_neighbour = sqrt(observations[1][0]**2 + observations[1][1]**2)
            if dist_to_neighbour <= self.crowd_radius_q:
                reward -= 3.0

            # OVO JE DA NE BUDE BLIZU PREPRECI
            # arr = observations[4]
            # for i in range(0, len(arr)):
            #     if sqrt(arr[i][0]**2 + arr[i][1]**2) <= self.avoid_radius_q:
            #         reward -=  5.0

            # OVO JE DA BUDE BLIZU JATU
            dist_to_flock = sqrt(observations[2][0]**2 + observations[2][1]**2)
            if dist_to_flock <= self.close_radius:
                reward += 10.0

            # OVO JE DA IMA ISTI SMJER
            steer_diff = abs(observations[0] - observations[3])
            reward -= steer_diff * 3.0
        else:
            reward += self.end_episode_points

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward


    # Internal TaskEnv Methods
    
    def discretize_observation(self, agent_steer, closest_agent_pose, direction, steer):#, closest_obst):
        """

        """
        self._episode_done = False
        if  not np.any(direction[:] == np.inf):

            # KVANTIZIRAJ SMJER AGENTA
            agent_steer = self.observation_quantization(agent_steer*180.0/np.pi, 0.0, 360.0, self.angle_quant_step)
            agent_steer = agent_steer * np.pi / 180.0

            # KVANTIZIRAJ POZICIJU NAJBLIZEG AGENTA I PREPREKA
            closest_agent_pose[0] = self.observation_quantization(closest_agent_pose[0], -self.rel_pose_max, self.rel_pose_max, self.quant_step)
            closest_agent_pose[1] = self.observation_quantization(closest_agent_pose[1], -self.rel_pose_max, self.rel_pose_max, self.quant_step)
            if sqrt(closest_agent_pose[0]**2 + closest_agent_pose[1]**2) <= self.too_close:
                self._episode_done = True

            # for i in range(0, len(closest_obst)):
            #     closest_obst[i][0] = self.observation_quantization(closest_obst[i][0], -self.rel_obst_pose_max, self.rel_obst_pose_max, self.quant_step)
            #     closest_obst[i][1] = self.observation_quantization(closest_obst[i][1], -self.rel_obst_pose_max, self.rel_obst_pose_max, self.quant_step)
            #     if sqrt(closest_obst[i][0]**2 + closest_obst[i][1]**2) <= self.too_close:
            #         self._episode_done = True

            # KVANTIZIRAJ POZICIJU I SMJER SUSJEDA
            direction[0] = self.observation_quantization(direction[0], -self.rel_pose_max, self.rel_pose_max, self.quant_step)
            direction[1] = self.observation_quantization(direction[1], -self.rel_pose_max, self.rel_pose_max, self.quant_step)

            steer = self.observation_quantization(steer*180.0/np.pi, 0.0, 360.0, self.angle_quant_step)
            steer = steer * np.pi / 180.0
            if steer == 2 * np.pi:
                steer = 0.0
        else:
            self._episode_done = True
            
        discretized_obs = (agent_steer, closest_agent_pose, direction, steer)#, closest_obst) # agent_pos        

        return discretized_obs

    def observation_quantization(self, value, min_value, max_value, qstep):
        if np.greater(value, max_value):
            value = max_value
        elif np.less(value, min_value):
            value = min_value

        return  qstep * np.round(value/qstep)