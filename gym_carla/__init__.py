from gym.envs.registration import register

# Torcs Custom
register(
	id="CarlaDefault-v0",
	entry_point='gym_carla.envs.default:DefaultEnv'
)
register(
	id="CarlaDefaultCustom-v0",
	entry_point='gym_carla.envs.default:DefaultEnvCustom'
)
