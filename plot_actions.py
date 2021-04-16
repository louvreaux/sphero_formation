import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
import json

def animate(i, fargs):
	lst =  fargs[i]
	x = range(len(lst))
	f_d.set_data(x, lst)
	ax.set_xlim(0, len(lst))
	temp.set_text('Episode: ' + str(i))

path = glob.glob('/home/user/catkin_ws/src/sphero_formation/training_results/openai_gym*') # You should change this to the name of your workspace
with open(path[0], 'r') as f:
	data = json.load(f)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_ylim(-0.5, 3.5)

f_d, = ax.plot([], [], 'bo')
temp = ax.text(0.1, 3.4, '', verticalalignment='top', horizontalalignment='left', fontsize=12)

actions_list = data['actions_by_episode']
ani = FuncAnimation(fig=fig, func=animate, fargs=(actions_list,), frames=len(actions_list), interval=1000, repeat=False)
plt.show()