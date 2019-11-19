# Gym Wrapper for Carla 0.9.6

## Dependencies and install

## Conda env install

## Basic usage
```python
import gym
import gym_carla

env = gym.make( 'CarlaDefault-v0', host='localhost', timestep_limit = 10000, render = True, render_mode = 'human')

for ep in range( 10):
  print( "Episode %d:" % ep)
  done = False
  obs = env.reset()

  # print( "# DEBUG: Done before entering episode", done)

  while not done:
    obs, _, done, _ = env.step( env.action_space.sample())
    print( 'Obs at timestep %d: %s' % (env._current_timestep, 'Not none' if obs is not None else 'None'))

    env.render()

env.close()
