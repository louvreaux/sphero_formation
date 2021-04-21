import json
import os
import time
import rospy

from gym import error
from gym.utils import atomic_write
from gym.utils.json_utils import json_encode_np

class StatsRecorder(object):
    def __init__(self, directory, autoreset=False, env_id=None):
        self.autoreset = autoreset
        self.env_id = env_id

        self.initial_reset_timestamp = None
        self.episode_lengths = []
        self.episode_rewards = []
        self.episode_types = [] # experimental addition
        self._type = 't'
        self.timestamps = []
        self.steps = None
        self.total_steps = 0
        self.rewards = None
        self.actions = []
        self.rewards_steps = []
        self.actions_episodes = []
        self.rewards_episodes = []

        self.done = None
        self.closed = False

        filename = 'openai_gym_batch.stats.json'
        self.path = os.path.join(directory, filename)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        if type not in ['t', 'e']:
            raise error.Error('Invalid episode type {}: must be t for training or e for evaluation', type)
        self._type = type

    def before_step(self, action):
        assert not self.closed

        self.actions.append(action)
        if self.done:
            raise error.ResetNeeded("Trying to step environment which is currently done. While the monitor is active for {}, you cannot step beyond the end of an episode. Call 'env.reset()' to start the next episode.".format(self.env_id))
        elif self.steps is None:
            raise error.ResetNeeded("Trying to step an environment before reset. While the monitor is active for {}, you must call 'env.reset()' before taking an initial step.".format(self.env_id))

    def after_step(self, observation, reward, done, info):
        self.steps += 1
        self.total_steps += 1
        self.rewards += reward
        self.rewards_steps.append(reward)
        self.done = done

        if done:
            self.save_complete()

        if done:
            if self.autoreset:
                self.before_reset()
                self.after_reset(observation)

    def before_reset(self):
        assert not self.closed

        if self.done is not None and not self.done and self.steps > 0:
            raise error.Error("Tried to reset environment which is not done. While the monitor is active for {}, you cannot call reset() unless the episode is over.".format(self.env_id))

        self.done = False
        if self.initial_reset_timestamp is None:
            self.initial_reset_timestamp = time.time()

    def after_reset(self, observation):
        self.steps = 0
        self.rewards = 0
        self.actions = []
        self.rewards_steps = []
        # We write the type at the beginning of the episode. If a user
        # changes the type, it's more natural for it to apply next
        # time the user calls reset().
        self.episode_types.append(self._type)

    def save_complete(self):
        if self.steps is not None:
            self.episode_lengths.append(self.steps)
            self.episode_rewards.append(float(self.rewards))
            self.timestamps.append(time.time())
            self.actions_episodes.append(self.actions)
            self.rewards_episodes.append(self.rewards_steps)

    def close(self, qlist):
        self.flush(qlist)
        self.closed = True

    def flush(self, qlist):
        if self.closed:
            return
        for key in qlist.keys():
            if type(key) is not str:
                try:
                    qlist[str(key)] = qlist[key]
                except:
                    try:
                        qlist[repr(key)] = qlist[key]
                    except:
                        pass
                del qlist[key]

        with atomic_write.atomic_write(self.path) as f:
            json.dump({
                'initial_reset_timestamp': self.initial_reset_timestamp,
                'timestamps': self.timestamps,
                'episode_lengths': self.episode_lengths,
                'episode_rewards': self.episode_rewards,
                'episode_types': self.episode_types,
                'actions_by_episode': self.actions_episodes,
                'rewards_by_episode': self.rewards_episodes,
                'qlearn_table': qlist,
            }, f, default=json_encode_np)
