# RL Sphero control
NOTICE: This is a work in progress.

This package directly adds onto: https://github.com/mkrizmancic/sphero_formation/tree/0ac14aad3dd1a0af26f191c017e213279eebd52e
It additonaly uses: 
1. _open\_ai\_ros_: http://wiki.ros.org/openai_ros
1. _DDPG-tf2_: https://github.com/samuelmat19/DDPG-tf2
1. _mantis\_ddqn\_navigation_: https://github.com/bhctsntrk/mantis_ddqn_navigation

## Dependencies
```
python 3.6
gym
tensorflow 2.x
xdotool
wmctrol
```

## About
This package connects Stage simulator with RL algorithms. It contains the following scripts:
1. _stage\_connection.py_ - class that contains _pause, unpause_ and _reset_ functions for Stage
1. _robot\_stage\_env.py_ - the most basic implementation of the RL _environment_
1. _sphero\_env.py_ - ROS-Sphero connection, inherits class from _robot\_stage\_env.py_
1. _sphero\_world.py_ - defines the task of the Sphero robot, inherits class from _sphero\_env.py_
1. _sphero\_qlearn.py_, _sphero\_dqn.py_, _sphero\_ddpg.py_ - RL algorithms

## Run
First of all, go to the package directory:
```bash
cd catkin_ws/src/sphero_formation
```
and initialize the updated _stage\_ros_:
```bash
git submodule update --init
```
Copy updated Stage to _/src_:
```bash
cp -r ~/catkin_ws/src/sphero_formation/stage_ros ~/catkin_ws/src
```
Everything else is the same as the _sphero\_formation_ package.
To change the RL algorithm, change to the appropriate script in _reynolds\_sim.launch_.
Also, change to the appropriate directories in order to save training results.

## Results

Simulation video for flocking using DDPG can be found [here](https://youtu.be/fip2qKlP3mo) and real Sphero robot [here](https://youtu.be/3JLKjcI3qBI).
