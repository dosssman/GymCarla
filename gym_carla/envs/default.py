import cv2
import gym
import time
import math
import carla
import pygame
import random
import numpy as np
from gym import spaces

# This should be the main dependecies
from gym_carla.envs.world_render_deps import World, HUD
from carla import ColorConverter as cc

# Required for user input interaction
# TODO: Remove possibility to control
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# Based on automatic_control example, good rendering method, custom observations (fixed)
class DefaultEnv( gym.Env):

    def __init__(self,
        # Carla Sim. Parameters
        host='localhost',
        port=2000,
        vehicle_bp='model3',

        # Env parameters
        timestep_limit = -1,
        render = False,
        render_mode = 'none', # Also accept agent vision
        display_width = 1280,
        display_height = 720,

        # Agent config
        agent_image_width = 640,
        agent_image_height = 480,
        agent_fov = 110,
        agent_actions_per_sec = 10,
        normalize_obs = True
        ):

        # Internal parameters
        self._host = host
        self._port = port
        self._render = render
        self._render_mode = render_mode
        # If won t render, then disable rendering down in the camera manager
        if self._render == False:
            self._render_mode = 'none'


        # Seed init, sued for random spawns
        self._seed = 42
        self.set_seed()

        # Human display config
        self._display_width = display_width
        self._display_height = display_height
        self._display_fov = 110 # TODO: Enable parametrization

        # Agent's camera config: Front view
        # TODO: Provide easy parametrization through gym
        self._image_width = agent_image_width
        self._image_height = agent_image_height
        self._fov = agent_fov
        self._normalize_obs = normalize_obs

        # Agent cation rate: how many actions per second
        # Worst case scenario: when rendering with human, the agent was empirically
        # measured to be able to do ~ 195. tsteps per second.
        # Hence, we set this value to stall the agent a bit. Therefore, sampling
        # when being evaluated while rendering, or without rendering should be ]
        # Approximately the same
        self._agent_actions_per_sec = agent_actions_per_sec

        # Agent collision monitor
        self._last_collision_frame = None
        # self._collisions_intensities = []

        # Gym Env config
        # TODO: Add observation spaces and other Gym Properties
        if self._normalize_obs:
            self.observation_space = spaces.Box(
                low=0., high=1., shape=(self._image_width, self._image_height, 3), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self._image_width, self._image_height, 3), dtype=np.uint8
            )

        self.action_space = spaces.Box(
            low=np.array([ -1., 0.]),
            high=np.array( [1., 1.]),
            dtype=float # TODO: How to Mix continuous and discrete
        )

        # State informations
        # The following to are now wrapped in CameraManager
        # self._current_observation = None # For agent observation and 'agent' mode rendering
        # self._current_display_observation = None # For the human rendering
        self._current_timestep = 0
        self._timestep_limit = timestep_limit
        self._done = False

        # Rendering related, TODO: Optimize following wether render = True or = False
        self._hud = None
        self._world = None
        self._display = None
        self._controller = None

        # Agent parameters ( Single Agent for now)
        self._vehicle_bp = vehicle_bp
        self._agent_vehicle_control = None

        # Initing some persistent objects
        self._agent_vehicle = None
        self._agent_front_camera = None
        self._display_camera = None

        # For reset and destroying later
        # self._actor_list = list()

        self._client = carla.Client( self._host, self._port)
        self._client.set_timeout( 2.0)

        self._agent_vehicle_control = carla.VehicleControl()

        self._agent_spwan_point = random.choice( self._client.get_world().get_map().get_spawn_points())

        world_args = {}
        world_args['vehicle_bp'] = self._vehicle_bp
        world_args['agent_spawn_point'] = self._agent_spwan_point
        world_args['render_mode'] = self._render_mode
        world_args['seed'] = self._seed

        # Agent camera sensor config
        camera_man_args = dict()

        camera_man_args['agent_image_width'] = self._image_width
        camera_man_args['agent_image_height'] = self._image_height
        camera_man_args['agent_fov'] = self._fov
        camera_man_args['render_mode'] = self._render_mode
        world_args['camera_man_args'] = camera_man_args

        # Rendering init
        if self._render:
            pygame.init()
            pygame.font.init()
            # TODO: Refactor
            if self._render_mode in ['human']:
                pygame_width = self._display_width
                pytgame_height = self._display_height

                if self._display is None:
                    self._display = pygame.display.set_mode(
                        (pygame_width, pytgame_height),
                        pygame.HWSURFACE | pygame.DOUBLEBUF
                    )


                if self._hud is None:
                    self._hud = HUD(self._display_width, self._display_height)

            elif self._render_mode in ['agent', 'agent_lidar']:
                pygame_width = self._image_width
                pytgame_height = self._image_height

                if self._display is None:
                    self._display = pygame.display.set_mode(
                        (pygame_width, pytgame_height),
                        pygame.HWSURFACE | pygame.DOUBLEBUF
                    )

            self._clock = pygame.time.Clock()

            # Display when not in agent render mode
            if self._render_mode in ['human']:
                camera_man_args['display_image_width'] = self._display_width
                camera_man_args['display_image_height'] = self._display_height
                camera_man_args['display_fov'] = self._display_fov

        self._world = World( self._client.get_world(), self._hud, world_args)
        self._world.world.wait_for_tick(10.0)

    def _reset(self):
        # First, destroy the existing agents
        # self._clean()

        # Reseting internal state
        self._current_timestep = 0
        self._done = False

        self._world.restart()

        self._agent_vehicle = self._world.player

        # as soon as the server is ready continue!
        # self._world.world.wait_for_tick(10.0)

        # self._world.camera_manager.toggle_recording()
        if self._render:
            self._world.tick(self._clock)
            self._world.render(self._display)

            pygame.display.flip()

        # Wait for the first observation to be loaded
        while True:
            if self._world.camera_manager._current_agent_observation is not None:
                break;

    def reset(self):
        self._reset()

        return self._world.camera_manager._current_agent_observation

    def _compute_reward(self):
        reward = 0.

        agent_velocity = self._agent_vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(agent_velocity.x**2 + agent_velocity.y**2
            + agent_velocity.z**2))

        # COllision penalization
        # print( "# DEBUGL: Collision history")
        # print( self._world.collision_sensor.history)

        if len(self._world.collision_sensor.history ):
            collisions_intensities = .0
            if self._last_collision_frame is None:
                self._last_collision_frame = self._world.collision_sensor.history[-1][0]
            else:
                if self._last_collision_frame < self._world.collision_sensor.history[-1][0]:
                    # Cumulate the latest collision data by comparing the frame: we don t want to
                    # repenalize for the same collision, right
                    collisions_intensities = sum(
                        [ collision_data[1] if collision_data[0] > self._last_collision_frame
                                           else .0  for collision_data in
                                           self._world.collision_sensor.history])

                    self._last_collision_frame = self._world.collision_sensor.history[-1][0]

            reward -= collisions_intensities

        # 1 Point if the agent maintains speed higher then 20
        if kmh > 20.:
            reward += 1
        # TODO: Reawrd promoting higher speed as a function of the latter itself

        return reward

    def step(self, action):
        if self._render:
            self._world.tick(self._clock)
            pygame.display.flip()


        self._agent_vehicle_control.steer = float( action[0])
        self._agent_vehicle_control.throttle = float( action[1])
        self._agent_vehicle.apply_control( self._agent_vehicle_control)

        self._current_timestep += 1

        # Terminsation condition
        if self._timestep_limit > 0 and self._current_timestep >= self._timestep_limit:
            self._done = True

        # if self._current_timestep % 200 == 0 and self._current_timestep > 0:
        # print( "# DEBUG: Current timestep: %d" % self._current_timestep)
        if self._agent_actions_per_sec > 0:
            time.sleep( 1 / self._agent_actions_per_sec)

        return self._current_observation(), self._compute_reward(), self._done, {}

    def _current_observation(self):
        if self._normalize_obs:
            return self._world.camera_manager._current_agent_observation / 255.
        else:
            return self._world.camera_manager._current_agent_observation

    def render(self):
        if self._render:
            self._world.render(self._display)

    def set_seed( self, seed = None):
        if seed is not None:
            self._seed = seed

        random.seed( self._seed)
        np.random.seed( self._seed)

    def close(self):
        if self._world is not None:
            self._world.destroy()

        pygame.quit()

# Env from scratch, rendering is amateur
class DefaultEnvCustom( gym.Env):

    def __init__(self,
        # Carla Sim. Parameters
        host='localhost',
        port=2000,
        vehicle_bp='model3',

        # Env parameters
        timestep_limit = -1,
        render = False,
        render_mode = 'human', # Also accept agent vision
        display_width = 1280,
        display_height = 720
        ):

        # Internal parameters
        self._host = host
        self._port = port
        self._render = render
        self._render_mode = render_mode # TODO: Control on values 'human', 'agent'
        self._display_width = display_width
        self._display_height = display_height
        self._display_fov = 110 # TODO: Enable parametrization
        self._seed = 42
        self.set_seed()

        # Gym Env config
        # TODO: Add observation spaces and other Gym Properties
        self.action_space = spaces.Box(
            low=np.array([ -1., 0.]),
            high=np.array( [1., 1.]),
            dtype=float # TODO: How to Mix continuous and discrete
        )

        # State informations
        self._current_observation = None # For agent observation and 'agent' mode rendering
        self._current_display_observation = None # For the human rendering
        self._current_timestep = 0
        self._timestep_limit = timestep_limit
        self._done = False

        # Agent config, ; TODO: Build it OOP style later
        self._vehicle_bp = vehicle_bp
        self._agent_vehicle_control = None

        # Agent's camera config: Front view
        self._image_width = 640
        self._image_height = 480
        self._fov = 110

        self._client = None

        # Simulator config
        self._client = carla.Client( self._host, self._port)
        self._client.set_timeout( 2.0)

        self._world = self._client.get_world()
        self._bp_library = self._world.get_blueprint_library() # Do we need to store it ?

        # Initing some persistent objects
        self._agent_vehicle = None
        self._agent_front_camera = None
        self._display_camera = None

        # For reset and destroying later
        self._actor_list = list()

        # Agent config
        self._agent_vehicle_bp = self._bp_library.filter( 'vehicle')
        self._agent_vehicle_bp = self._agent_vehicle_bp.filter( self._vehicle_bp)[0]
        self._agent_spwan_point = random.choice( self._world.get_map().get_spawn_points())

        # Agent camera config
        self._front_camera_bp = self._bp_library.find( 'sensor.camera.rgb')
        self._front_camera_bp.set_attribute( 'image_size_x', str( self._image_width))
        self._front_camera_bp.set_attribute( 'image_size_y', str( self._image_height))
        self._front_camera_bp.set_attribute( 'fov', str( self._fov))
        self._front_camera_transform = carla.Transform( carla.Location( x=1.5, z=2.0))

        # Display camere init
        # TODO: Refrain from creatin if doesn t plan on rendering ?
        self._display_camera_bp = self._bp_library.find( 'sensor.camera.rgb')
        self._display_camera_bp.set_attribute( 'image_size_x', str( self._display_width))
        self._display_camera_bp.set_attribute( 'image_size_y', str( self._display_height))
        self._display_camera_bp.set_attribute( 'fov', str( self._display_fov))
        self._display_camera_transform = carla.Transform( carla.Location( x=-5.5, z=2.8))

        self._agent_vehicle_control = carla.VehicleControl()

        self._reset()

    def _reset(self):
        # First, destroy the existing agents
        self._clean()

        self._current_timestep = 0
        self._current_observation = None
        self._current_display_observation = None
        self._done = False

        # (Re Instantiating agent)
        # Agent doesn 't exist, creating new one
        self._agent_vehicle = self._world.try_spawn_actor( self._agent_vehicle_bp,
            self._agent_spwan_point)

        # DEBUG: Set autopilot to skip step( action)
        self._actor_list.append( self._agent_vehicle)

        # Agent's front camera
        self._agent_front_camera = self._world.try_spawn_actor( self._front_camera_bp,
            self._front_camera_transform, attach_to=self._agent_vehicle)

        self._actor_list.append( self._agent_front_camera)

        self._display_camera = self._world.try_spawn_actor( self._display_camera_bp,
            self._display_camera_transform, attach_to=self._agent_vehicle)

        self._actor_list.append( self._display_camera)

        # register hooks to feed the observations
        self._agent_front_camera.listen( lambda image: self._process_agent_front_cam( image))
        self._display_camera.listen( lambda image: self._process_display_cam( image))

        while True:
            # TODO: When not rendering, just confition on self._current_observation
            if self._current_observation is not None and self._current_display_observation is not None:
                break;

    def reset(self):
        self._reset()

        # TODO: Add waiting period time to make sure self._current_observation is loaded
        return self._current_observation

    def step( self, action):
        self._agent_vehicle_control.steer = float( action[0])
        self._agent_vehicle_control.throttle = float( action[1])
        self._agent_vehicle.apply_control( self._agent_vehicle_control)

        self._current_timestep += 1

        # TODO: Add score computation function
        if self._timestep_limit > 0 and self._current_timestep > self._timestep_limit:
            self._done = True

        return self._current_observation, 0., self._done, {}

    def render(self):
        if self._render_mode == 'human':
            cv2.imshow( '', self._current_display_observation)
            cv2.waitKey( 33)
        elif self._render_mode == 'agent':
            cv2.imshow( '', self._current_observation)
            cv2.waitKey( 33)
        else:
            raise NotImplementedError

    def _process_agent_front_cam( self, image):
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, ( self._image_height, self._image_width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._current_observation = array

    def _process_display_cam( self, image):
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, ( self._display_height, self._display_width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._current_display_observation = array

    def _clean( self):
        print( "# DEBUG: Destroying actors -- Actor count: %d" % (len( self._actor_list)))
        for actor in self._actor_list:
            actor.destroy()

        self._actor_list = list()

    def close(self):
        self._clean()

    def set_seed( self, seed = None):
        if seed is not None:
            self._seed = seed

        random.seed( self._seed)
        np.random.seed( self._seed)
