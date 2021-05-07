import rospy
import random
import numpy as np

from gym import spaces
from gym.envs.registration import register
from geometry_msgs.msg import Vector3, Twist
from math import sqrt, cos, sin
from robot_envs import sphero_env

#timestep_limit_per_episode = 2500 # Can be any Value

register(
		id='SpheroWorld-v1',
		entry_point='task_envs.sphero.sphero_world1:SpheroWorldEnv',
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
		self.action_space = spaces.Discrete(int(360.0/self.angle_quant_step))

		self.observation_space = spaces.Box(low=np.array([-2*np.pi, -self.rel_pose_max, -self.rel_pose_max, -self.rel_pose_max, -self.rel_pose_max]), 
											high=np.array([2*np.pi, self.rel_pose_max, self.rel_pose_max, self.rel_pose_max, self.rel_pose_max]), 
											dtype=np.float64)

		# We set the reward range, which is not compulsory but here we do it.
		self.reward_range = (-np.inf, np.inf)
		
		
		rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
		rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

		self.cumulated_steps = 0.0
		self.last_obs = ()

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
		agent_steer, closest_neighbour, flock_pose, flock_steer, closest_obst, num_of_neighbours = self.get_callback()

		if num_of_neighbours > 0:
			self._episode_done = False

			agent_steer = agent_steer * 180.0/np.pi
			flock_steer = self.observation_quantization(flock_steer * 180.0/np.pi, 0.0, 360.0, self.angle_quant_step)
			if flock_steer == 360.0:
				flock_steer = 0.0
			steer_angle_diff = agent_steer - flock_steer

			if sqrt(closest_agent_pose[0]**2 + closest_agent_pose[1]**2) <= self.too_close:
				self._episode_done = True
				# self.crash = True

			observations = np.array([steer_angle_diff, closest_agent_pose[0], closest_agent_pose[1], flock_pose[0], flock_pose[1]])
			self.last_obs = observations
		else:
			self._episode_done = True
			observations = self.last_obs
		rospy.logdebug("Observations==>"+str(observations))
		rospy.logdebug("END Get Observation ==>")
		rospy.logwarn(observations)

		return observations
		

	def _is_done(self, observations):
		
		if self._episode_done:
			rospy.logerr("Sphero has diverged from flock or has hit something!")

		return self._episode_done

	def _compute_reward(self, observations, done):

		reward = 0.0

		if not done:

			# OVO JE DA NE BUDE BLIZU NAJBLIZEM AGENTU
			dist_to_neighbour = sqrt(observations[1][0]**2 + observations[1][1]**2)
			if dist_to_neighbour <= 0.3 and dist_to_neighbour > 0.1:
				reward += 25 * dist_to_neighbour - 7.5
			elif dist_to_neighbour <= 0.1:
				reward -= 5

			# OVO JE DA NE BUDE BLIZU PREPRECI
			# arr = observations[4]
			# for i in range(0, len(arr)):
			#     if sqrt(arr[i][0]**2 + arr[i][1]**2) <= self.avoid_radius_q:
			#         reward -=  5.0

			# OVO JE DA BUDE BLIZU JATU
			dist_to_flock = sqrt(observations[2][0]**2 + observations[2][1]**2)
			#if dist_to_flock <= 0.8 and dist_to_flock > 0.6:
			#    reward -= 5
			if dist_to_flock <= 0.6 and dist_to_flock > 0.1:
				reward += -10.0 * dist_to_flock + 6.0
			elif dist_to_flock <= 0.1:
				reward += 5

			# OVO JE DA IMA ISTI SMJER
			if observations[0] != 0.0:
				reward -= 3.0
		else:
			 reward += self.end_episode_points

		rospy.logdebug("reward=" + str(reward))
		self.cumulated_reward += reward
		rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
		self.cumulated_steps += 1
		rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
		
		return reward

	def observation_quantization(self, value, min_value, max_value, qstep):
		if np.greater(value, max_value):
			value = max_value
		elif np.less(value, min_value):
			value = min_value

		return  qstep * np.round(value/qstep)