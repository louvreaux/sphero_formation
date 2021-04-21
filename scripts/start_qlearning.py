#!/usr/bin/env python

import gym
import numpy
import time
import qlearn
from stats_recorder import StatsRecorder
# ROS packages required
import rospy
import rospkg

from task_envs.sphero import sphero_world
from sphero_formation.msg import OdometryArray
from nav_msgs.msg import Odometry

if __name__ == '__main__':

    rospy.init_node('sphero_qlearn', anonymous=True, log_level=rospy.ERROR) #log_level=rospy.DEBUG
    r = rospy.Rate(10)

    # Init OpenAI_ROS ENV
    env = gym.make('SpheroWorld-v0')
    # Create the Gym environment
    rospy.loginfo("Gym environment done")
    rospy.loginfo("Starting Learning")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('sphero_formation')
    outdir = pkg_path + '/training_results'
    #env = wrappers.Monitor(env, outdir, force=True)
    sr = StatsRecorder(outdir)
    rospy.loginfo("Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    Alpha = rospy.get_param("alpha")
    Epsilon = rospy.get_param("epsilon")
    Gamma = rospy.get_param("gamma")
    epsilon_discount = rospy.get_param("epsilon_discount")
    nepisodes = rospy.get_param("nepisodes")
    nsteps = rospy.get_param("nsteps")

    running_step = rospy.get_param("running_step")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                           alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0
    
    temp_msg = Odometry()
    while abs(temp_msg.twist.twist.linear.y) <= 0.05:
        temp_msg = rospy.wait_for_message('/robot_0/odom', Odometry)

    try:
        # Starts the main training loop: the one about the episodes to do
        for x in range(nepisodes):
            rospy.logdebug("############### START EPISODE=>" + str(x))

            cumulated_reward = 0
            done = False
            if qlearn.epsilon > 0.05:
                qlearn.epsilon *= epsilon_discount

            # Initialize the environment and get first state of the robot
            sr.before_reset()
            observation = env.reset()
            sr.after_reset(observation)
            state = ''.join(map(str, observation))

            # Show on screen the actual situation of the robot
            # env.render()
            # for each episode, we test the robot for nsteps
            for i in range(nsteps):
                rospy.logwarn("############### Start Step=>" + str(i))
                # Pick an action based on the current state
                action = qlearn.chooseAction(state)
                rospy.logdebug("Next action is:%d", action)
                # Execute the action in the environment and get feedback
                sr.before_step(action)
                observation, reward, done, info = env.step(action)
                sr.after_step(observation, reward, done, info)

                rospy.logdebug(str(observation) + " " + str(reward))
                cumulated_reward += reward
                if highest_reward < cumulated_reward:
                    highest_reward = cumulated_reward

                nextState = ''.join(map(str, observation))

                # Make the algorithm learn based on the results
                rospy.logdebug("# state we were=>" + str(state))
                rospy.logdebug("# action that we took=>" + str(action))
                rospy.logdebug("# reward that action gave=>" + str(reward))
                rospy.logdebug("# episode cumulated_reward=>" + str(cumulated_reward))
                rospy.logdebug("# State in which we will start next step=>" + str(nextState))
                qlearn.learn(state, action, reward, nextState)

                if not (done):
                    rospy.logdebug("NOT DONE")
                    state = nextState
                else:
                    rospy.logdebug("DONE")
                    last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                    break
                rospy.logwarn("############### END Step=>" + str(i))
                #raw_input("Next Step...PRESS KEY")
                r.sleep()
            m, s = divmod(int(time.time() - start_time), 60)
            h, m = divmod(m, 60)
            rospy.logerr(("EP: " + str(x + 1) + " - [alpha: " + str(round(qlearn.alpha, 2)) + " - gamma: " + str(
                round(qlearn.gamma, 2)) + " - epsilon: " + str(round(qlearn.epsilon, 2)) + "] - Reward: " + str(
                cumulated_reward) + "     Time: %d:%02d:%02d" % (h, m, s)))

        rospy.loginfo(("\n|" + str(nepisodes) + "|" + str(qlearn.alpha) + "|" + str(qlearn.gamma) + "|" + str(
            initial_epsilon) + "*" + str(epsilon_discount) + "|" + str(highest_reward) + "| PICTURE |"))

        l = last_time_steps.tolist()
        l.sort()

        # print("Parameters: a="+str)
        rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
        rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

        sr.close()
        env.close()

    except rospy.ROSInterruptException:
        sr.close(qlearn.q)
        env.close()