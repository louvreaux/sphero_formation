import rospy
import random
import numpy as np

from gym import spaces
from gym.envs.registration import register
from geometry_msgs.msg import Twist
from math import cos, sin
from robot_envs import sphero_env

#timestep_limit_per_episode = 2500 # Can be any Value

register(
		id='SpheroWorld-v2',
		entry_point='task_envs.sphero.sphero_world2:SpheroWorldEnv',
		#max_episode_steps=timestep_limit_per_episode,
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
		# self.action_space = spaces.Box(low=np.array([-1*np.pi]), high=np.array([np.pi]), dtype=np.float64)
		self.action_space = spaces.Box(low=np.array([-1*np.pi, 0.0]), high=np.array([np.pi, 0.3]), dtype=np.float64)

		self.observation_space = spaces.Box(low=np.array([-1*np.pi, 0.0, 0.0, 0.0, 0.0, -0.3]), 
											high=np.array([np.pi, 2*np.pi, self.rel_pose_max, 2*np.pi, self.rel_pose_max, 0.3]), 
											dtype=np.float64)

		# We set the reward range, which is not compulsory but here we do it.
		self.reward_range = (-np.inf, np.inf)
		
		
		rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
		rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

		self.cumulated_steps = 0.0

		# Here we will add any init functions prior to starting the MyRobotEnv
		super(SpheroWorldEnv, self).__init__()
		
		# Init first state while starting learning
		self.first_state = True
		self.crash = False


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

		# We convert the actions to speed movements

		# ONLY ONE ACTION PARAMETER -> diff_angle
		# new_vel = np.array([cos(action), sin(action)]) * self.vel_max # We rotate vector [1, 0] to the left

		# TWO ACTION PARAMETERS -> [diff_angle, speed]
		new_vel = np.array([cos(action[0]), sin(action[0])]) * action[1]
		
		# rospy.logerr("ACTION: " + str(action))
		# rospy.logerr("ACTION ANGLE: " + str(theta * 180.0 / np.pi))
		
		# Publish velocity
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

		steer_diff, closest_neighbour, flock_pose, flock_steer, vel_diff, closest_obst, num_of_neighbours = self.get_callback()

		# If our agent didn't loose the flock, examine observations
		if num_of_neighbours > 0:
			self._episode_done = False

			# # We discretize steering because of discrete actions
			# steer_diff = self.observation_quantization(steer_diff * 180.0/np.pi, -180.0, 180.0, self.angle_quant_step)
			# steer_diff = steer_diff * np.pi / 180.0
			# if abs(steer_diff) <= 0.001:
			# 	steer_diff = 0.0

			# If agent is too close to neighbour, finish episode
			if closest_neighbour[1] <= self.too_close:
				self.crash = True
				self._episode_done = True

			observations = np.array([steer_diff, closest_neighbour[0], closest_neighbour[1], flock_pose[0], flock_pose[1], vel_diff])
			self.last_obs = observations

		# Else, return last known observation
		else:
			self._episode_done = True
			observations = self.last_obs
		
		rospy.logdebug("Observations==>"+str(observations))
		rospy.logdebug("END Get Observation ==>")
		
		# rospy.logerr("FLOCK ANGLE: " + str(flock_steer * 180.0 / np.pi))
		# rospy.logerr("STEER DIFF: " + str(steer_diff * 180.0 / np.pi))
		# rospy.logerr("OBSERVATIONS: " + str(observations))

		return observations
		

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
			# if observations[2] <= 0.3 and observations[2] > 0.1:
			# 	reward += 25 * observations[2] - 7.5
			# elif observations[2] <= 0.1:
			# 	reward -= 5
			reward += (5.0 / (1.0 + np.exp(-50.0 * (observations[2] - 0.2))) - 5.0)
			# temp = reward
			# rospy.logerr("CLOSEST NEIGHBOUR REWARD: " + str(reward))

			# Punish being close to an obstacle
			# arr = observations[4]
			# for i in range(0, len(arr)):
			#     if sqrt(arr[i][0]**2 + arr[i][1]**2) <= self.avoid_radius_q:
			#         reward -=  5.0

			# Reward being close to flock
			# if observations[4] <= 0.6 and observations[4] > 0.1:
			# 	reward += -10.0 * observations[4] + 6.0
			# elif observations[4] <= 0.1:
			# 	reward += 5
			reward += (5.0 - 5.0 / (1.0 + np.exp(-20.0 * (observations[4] - 0.35))))
			# rospy.logerr("FLOCK REWARD: " + str(reward - temp))
			# temp = reward

			# Reward going the similar way as flock
			# reward += 5.0 - 5.0 / np.pi * abs(observations[0])
			reward += (3.0 - 3.0 / (1.0 + np.exp(-4.0 * (abs(observations[0] - np.pi/2.0)))))
			# rospy.logerr("STEER REWARD: " + str(reward - temp))

			reward += (2.0 - 2.0 / (1.0 + np.exp(-50.0 * (abs(observations[5]) - 0.125))))
		else:
			 reward -= 10.0

		rospy.logdebug("reward=" + str(reward))
		self.cumulated_reward += reward
		rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
		self.cumulated_steps += 1
		rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
		
		return reward

	def observation_quantization(self, value, min_value, max_value, qstep):
		""" Used for discretization of observations, usually needed for
			discrete algorithms, such as q-learning """
		if np.greater(value, max_value):
			value = max_value
		elif np.less(value, min_value):
			value = min_value

		return  qstep * np.round(value/qstep)
