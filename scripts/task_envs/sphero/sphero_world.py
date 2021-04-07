import rospy
import numpy as np

from gym import spaces
from gym.envs.registration import register
from geometry_msgs.msg import Vector3, Twist
from math import sqrt, cos, sin
from robot_envs import sphero_env

# The path is __init__.py of openai_ros, where we import the TurtleBot2MazeEnv directly
timestep_limit_per_episode = 10000 # Can be any Value

register(
        id='SpheroWorld-v0',
        entry_point='task_envs.sphero.sphero_world:SpheroWorldEnv',
        max_episode_steps=timestep_limit_per_episode,
    )

class SpheroWorldEnv(sphero_env.SpheroEnv):
    def __init__(self):
        # Load needed parameters
        self.quant_step = rospy.get_param('quantization_step')

        self.vel_max = rospy.get_param('max_velocity')
        self.pose_max = rospy.get_param('max_pose')
        self.rel_pose_max = rospy.get_param('max_relative_pose')
        self.rel_obst_pose_max = rospy.get_param('max_relative_pose_to_obstacle')

        self.too_close = rospy.get_param('too_close')
        self.crowd_radius_q = rospy.get_param('crowd_radius_q')
        self.avoid_radius_q = rospy.get_param('avoid_radius_q')
        self.close_radius = rospy.get_param('close_radius')

        self.end_episode_points = rospy.get_param('end_episode_points')


        # Only variable needed to be set here
        # self.action_space = spaces.Box(low = -self.vel_max, high = self.vel_max, shape = (1,2), dtype = np.float64)
        self.action_space = spaces.Discrete(int(360.0/self.quant_step+1))

        self.observation_space = spaces.Tuple((
                    spaces.Box(low = -self.pose_max, high = self.pose_max, shape = (1,2), dtype = np.float64),                          # pozicija agenta
                    spaces.Box(low = -self.vel_max, high = self.vel_max, shape = (1,2),  dtype = np.float64),                           # brzina agenta
                    spaces.Box(low = -self.rel_pose_max, high = self.rel_pose_max, shape = (1,2), dtype = np.float64),                  # najblizi agent
                    spaces.Box(low = -self.rel_pose_max, high = self.rel_pose_max, shape = (1,2), dtype = np.float64),                  # pozicija grupiranja
                    spaces.Box(low = -self.vel_max, high = self.vel_max, shape = (1,2), dtype = np.float64),                            # brzina kretanja jata
                    spaces.Box(low = -self.rel_obst_pose_max, high = self.rel_obst_pose_max, shape = (10, 2), dtype = np.float64)))      # pozicije prepreka

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, 0.0)
        self.new_vel = np.array([self.vel_max, 0])
        
        
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
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        theta = action * self.quant_step * (np.pi/180.0)
        self.new_vel = np.array([self.new_vel[0] * cos(theta) - self.new_vel[1] * sin(theta),
                                 self.new_vel[0] * sin(theta) + self.new_vel[1] * cos(theta)])
        
        # We tell TurtleBot2 the linear and angular speed to set to execute
        send_velocity = Twist()
        send_velocity.linear.x = self.new_vel[0]
        send_velocity.linear.y = self.new_vel[1]
        self.move_base(send_velocity)
        
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        # We get the laser scan data
        agent_pos, agent_vel, closest_agent_pose, direction, steer, closest_obst = self.get_callback()
        
        discretized_observations = self.discretize_observation(agent_pos, agent_vel, closest_agent_pose, direction, steer, closest_obst)
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

        if not done:

            reward = 0.0

            # OVO JE DA NE BUDE BLIZU NAJBLIZEM AGENTU
            dist_to_neighbour = observations[0] - observations[2]
            dist_to_neighbour = sqrt(dist_to_neighbour[0]**2 + dist_to_neighbour[1]**2)
            if dist_to_neighbour <= self.crowd_radius_q:
                reward -= 3.0 # 1.0/dist_to_neighbour

            # OVO JE DA NE BUDE BLIZU PREPRECI
            arr = observations[5]
            for i in range(0, len(arr)):
                if (arr[i][0]**2 + arr[i][1]**2 <= self.avoid_radius_q**2):
                    reward -=  5.0 # 1.0/(sqrt(arr[i][0]**2 + arr[i][1]**2))

            # OVO JE DA BUDE BLIZU JATU
            dist_to_flock = observations[0] - observations[3]
            dist_to_flock = sqrt(dist_to_flock[0]**2 + dist_to_flock[1]**2)
            if dist_to_flock >= self.close_radius:
                reward -= 3.0 # 1.0/dist_to_flock

            # OVO JE DA IMA ISTI SMJER
            steer_diff = observations[1] - observations[4]
            steer_diff = sqrt(steer_diff[0]**2 + steer_diff[1]**2)
            reward -= steer_diff * 3.0
        else:
            reward = self.end_episode_points

        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        
        return reward


    # Internal TaskEnv Methods
    
    def discretize_observation(self, agent_pos, agent_vel, closest_agent_pose, direction, steer, closest_obst):
        """
        Discards all the laser readings that are not multiple in index of new_ranges
        value.
        """
        self._episode_done = False
        if  not np.any(direction[:] == np.inf):

            # KVANTIZIRAJ POZICIJIU I BRZINU AGENTA
            agent_pos[0] = self.observation_quantization(agent_pos[0], -self.pose_max, self.pose_max, self.quant_step)
            agent_pos[1] = self.observation_quantization(agent_pos[1], -self.pose_max, self.pose_max, self.quant_step)
            agent_vel[0] = self.observation_quantization(agent_vel[0], -self.vel_max, self.vel_max, self.quant_step)
            agent_vel[1] = self.observation_quantization(agent_vel[1], -self.vel_max, self.vel_max, self.quant_step)

            # KVANTIZIRAJ POZICIJU NAJBLIZEG AGENTA I PREPREKA
            closest_agent_pose[0] = self.observation_quantization(closest_agent_pose[0], -self.rel_pose_max, self.rel_pose_max, self.quant_step)
            closest_agent_pose[1] = self.observation_quantization(closest_agent_pose[1], -self.rel_pose_max, self.rel_pose_max, self.quant_step)
            if (closest_agent_pose[0]**2 + closest_agent_pose[1]**2 <= self.too_close**2):
                self._episode_done = True

            for i in range(0, len(closest_obst)):
                closest_obst[i][0] = self.observation_quantization(closest_obst[i][0], -self.rel_obst_pose_max, self.rel_obst_pose_max, self.quant_step)
                closest_obst[i][1] = self.observation_quantization(closest_obst[i][1], -self.rel_obst_pose_max, self.rel_obst_pose_max, self.quant_step)
                if (closest_obst[i][0])**2 + (closest_obst[i][1])**2 <= (self.too_close)**2:
                    self._episode_done = True

            # KVANTIZIRAJ POZICIJU I BRZINU SUSJEDA
            direction[0] = self.observation_quantization(direction[0], -self.rel_pose_max, self.rel_pose_max, self.quant_step)
            direction[1] = self.observation_quantization(direction[1], -self.rel_pose_max, self.rel_pose_max, self.quant_step)
            steer[0] = self.observation_quantization(steer[0], -self.vel_max, self.vel_max, self.quant_step)
            steer[1] = self.observation_quantization(steer[1], -self.vel_max, self.vel_max, self.quant_step)
        else:
            self._episode_done = True
            
        discretized_obs = (agent_pos, agent_vel, closest_agent_pose, direction, steer, closest_obst)            

        return discretized_obs

    def observation_quantization(self, value, min_value, max_value, step):
        if np.greater(value, max_value):
            value = max_value
        elif np.less(value, min_value):
            value = min_value

        return  step * np.round(value/step)