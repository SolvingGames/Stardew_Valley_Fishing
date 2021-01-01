

# libary for reinforcement learning games
import gym
# stardew valley environment class importing like this to avoid error
# https://github.com/mpSchrader/gym-sokoban/issues/29
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
     if 'sv-v0' in env:
          print('Remove {} from registry'.format(env))
          del gym.envs.registration.registry.env_specs[env]
import gym_sv
# local file to send inputs to the game
import inputclasses

# https://github.com/aborghi/retro_contest_agent/blob/master/fastlearner/ppo2ttifrutti_sonic_env.py
import numpy as np
# stacking frames
from baselines.common.atari_wrappers import FrameStack
# modify frames (former times we used matplotlib)
import cv2
# setUseOpenCL = False means that we will not use GPU (disable OpenCL acceleration)
cv2.ocl.setUseOpenCL(False)

class PreprocessFrame(gym.ObservationWrapper):
    """
    Here we do the preprocessing part:
    - Set frame to gray
    - Resize the frame to 96x96x1
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        # Resize the frame to 96x96x1
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]
        return frame


class ActionsDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the game.
    """
    def __init__(self, env):
        super(ActionsDiscretizer, self).__init__(env)
        buttons = ["C", "nothing"]
        actions = [["C"], ['nothing']]
        self._actions = []

        for action in actions:
            _actions = np.array([False] * 2) #len[buttons]
            for button in action:
                _actions[buttons.index(button)] = True
            self._actions.append(_actions)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):

        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

def make_env():
    """
    Create an environment with some standard wrappers.
    """
    # make the environment
    env = gym.make('sv-v0')
    #env = gym.make('AirRaid-v0')
    # Build the actions array, 
    env = ActionsDiscretizer(env)
    # Scale the rewards
    env = RewardScaler(env)
    # PreprocessFrame
    env = PreprocessFrame(env)
    # Stack 4 frames
    env = FrameStack(env, 4)
    # allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
    env = AllowBacktracking(env)

    return env


if __name__ == '__main__':
    import time
    env = make_env()
    env.reset()
    now = time.time()
    print('Executing this file should send inputs to stardew valley for a short period of time.')
    for _ in range(100):
        #env.render()
        env.step(env.action_space.sample()) # take a random action
        #print('working for 100 steps...')
    now = time.time() - now
    print('Frames per second: ',100/now)
    env.close()