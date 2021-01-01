from gym.envs.registration import register

register(
    id='sv-v0',
    entry_point='gym_sv.envs:svEnv',
)