import svgymwrapped
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv


env = svgymwrapped.make_env()
env = DummyVecEnv([lambda: env])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=128)
model.save("ppo2")

'''
for _ in range(10):
    svgymwrapped.make_env().reset()
    model.learn(total_timesteps=512)

model.save("ppo2")
'''