import svgymwrapped
from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
import time

import cv2
cv2.destroyAllWindows()

FILENAME = "ppo2"


time.sleep(2)
#load model from file
model = PPO2.load(FILENAME)
#create environment
env = svgymwrapped.make_env()
#wrap environment
env = DummyVecEnv([lambda: env])
#set environment to model
model.set_env(env)


model = PPO2.load("ppo2")
# Enjoy trained agent
obs = env.reset()
for _ in range(300):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
