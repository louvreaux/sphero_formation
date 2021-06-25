import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import glob
import json

def animate(i, *fargs):
	ac_lst =  fargs[0][i]
	rw_lst = fargs[1][i]
	x = range(len(ac_lst))

	f_d.set_data(x, ac_lst)
	f_d2.set_data(x, rw_lst)

	ax.set_xlim(0, len(ac_lst))
	ax2.set_xlim(0, len(ac_lst))
	ax2.set_ylim(min(rw_lst), max(rw_lst))

	temp.set_text('Episode: ' + str(i+1))

path = glob.glob('/home/user/catkin_ws/src/sphero_formation/training_results/openai_gym*')
with open(path[0], 'r') as f:
	data = json.load(f)

fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax.set_ylim(-0.5, 3.5)

f_d, = ax.plot([], [], 'bo')
f_d2, = ax2.plot([], [], 'ro')
temp = ax.text(0.1, 3.4, '', verticalalignment='top', horizontalalignment='left', fontsize=12)

actions_list = data['actions_by_episode']
rewards_list = data['rewards_by_episode']
ani = FuncAnimation(fig=fig, func=animate, fargs=(actions_list, rewards_list), frames=len(actions_list), interval=1000, repeat=False)
plt.show()