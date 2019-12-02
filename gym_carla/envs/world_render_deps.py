import re
import math
import time
import carla
import random
import pygame
import weakref
import datetime
import collections
import numpy as np

from carla import ColorConverter as cc

# Helpers
# TODO: Separate into different lib

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================
class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        if self.hud is not None:
            self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 1000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        if self.hud is not None:
            self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================
class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraManager(object):
    def __init__(self, parent_actor, hud, camera_man_args):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), # Third Person View
            carla.Rotation(pitch=-15)), # ?
            carla.Transform(carla.Location(x=1.6, z=1.7)) # First Person View
        ]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']
        ]
        # dosssman custom
        # allow external advanced configs
        self._camera_man_args = camera_man_args
        # TODO: add parmetrization
        self._render_mode = camera_man_args['render_mode']

        # Passing the ref for the current observation to be written into the env directy
        self._current_agent_observation = None
        self._current_lidar_observation = None
        self._current_display_observation = None

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()

        # TODO: Clean up this once familiar with sensor setting process
        # for item in self.sensors:
        #     bp = bp_library.find(item[0])
        #     if item[0].startswith('sensor.camera'):
        #         bp.set_attribute('image_size_x', str(hud.dim[0]))
        #         bp.set_attribute('image_size_y', str(hud.dim[1]))
        #     elif item[0].startswith('sensor.lidar'):
        #         bp.set_attribute('range', '5000')
        #     item.append(bp)
        self.index = None

        # Custom: Creating agent s front view
        agent_front_camera_bp = bp_library.find( 'sensor.camera.rgb')
        agent_front_camera_bp.set_attribute( 'image_size_x', str( self._camera_man_args['agent_image_width']))
        agent_front_camera_bp.set_attribute( 'image_size_y', str( self._camera_man_args['agent_image_height']))
        agent_front_camera_bp.set_attribute( 'fov', str( self._camera_man_args['agent_fov']))

        # Avoid circular ref later when processing images
        weak_self = weakref.ref(self)

        self._agent_front_cam_sensor = self._parent.get_world().spawn_actor(
            agent_front_camera_bp,
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            attach_to=self._parent
        )

        self._agent_front_cam_sensor.listen( lambda image: CameraManager._parse_front_cam_image(weak_self, image))

        # Do not create third person view sensor if human is not looking.
        if self._render_mode == 'human':
        # Camera for the human, TPS view of the agent
            display_camera_bp = bp_library.find( 'sensor.camera.rgb')
            display_camera_bp.set_attribute( 'image_size_x', str( self._camera_man_args['display_image_width']))
            display_camera_bp.set_attribute( 'image_size_y', str( self._camera_man_args['display_image_height']))
            display_camera_bp.set_attribute( 'fov', str( self._camera_man_args['display_fov']))

            self._display_cam_sensor = self._parent.get_world().spawn_actor(
                display_camera_bp,
                carla.Transform(carla.Location(x=-5.5, z=2.8)),
                attach_to=self._parent
            )
            self._display_cam_sensor.listen( lambda image: CameraManager._parse_display_cam_image(weak_self, image))

        # Lidar sensor. Skip it for now. Using pics only
        # agent_lidar_bp = bp_library.find( 'sensor.lidar.ray_cast')
        # agent_lidar_bp.set_attribute( 'range', str( '15'))

        # self._agent_lidar_sensor = self._parent.get_world().spawn_actor(
        #     agent_lidar_bp,
        #     carla.Transform(carla.Location(x=.0, z=.0)),
        #     attach_to=self._parent
        # )
        # self._agent_lidar_sensor.listen( lambda data: CameraManager._parse_lidar_data( weak_self, data))

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    # TODO: Delete as not used for RL
    # def set_sensor(self, index, notify=True):
    #     index = index % len(self.sensors)
    #     needs_respawn = True if self.index is None \
    #         else self.sensors[index][0] != self.sensors[self.index][0]
    #     if needs_respawn:
    #         if self.sensor is not None:
    #             self.sensor.destroy()
    #             self.surface = None
    #         self.sensor = self._parent.get_world().spawn_actor(
    #             self.sensors[index][-1],
    #             self._camera_transforms[self.transform_index],
    #             attach_to=self._parent)
    #         # We need to pass the lambda a weak reference to self to avoid
    #         # circular reference.
    #         weak_self = weakref.ref(self)
    #         self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
    #     if notify:
    #         self.hud.notification(self.sensors[index][2])
    #     self.index = index

    # def next_sensor(self):
    #     self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    # TODO: Delete because not used ?
    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            print( "# DEBUG: Saving img to disk")
            image.save_to_disk('_out/%08d' % image.frame)

    @staticmethod
    def _parse_display_cam_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        # TODO: Still need to parametrize here depending on what display
        # the user actually wants to see
        # self._current_display_observation = array
        if self._render_mode == 'human':
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        # TODO: Come back here for recording later
        if self.recording:
            print( "# DEBUG: Saving img to disk")
            image.save_to_disk('_out/%08d' % image.frame)

    @staticmethod
    def _parse_front_cam_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        # array = np.reshape(array, (image.height, image.width, 4))

        array = array[:, :, :3]
        array = array[:, :, ::-1]

        # TODO: Visualize this and make sure it actually reflects the pic properly
        self._current_agent_observation = np.reshape( array, (image.width, image.height, 3))

        # Do we render this some day ?
        if self._render_mode == 'agent':
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        # TODO: Come back here for recording later
        if self.recording:
            print( "# DEBUG: Saving img to disk")
            image.save_to_disk('_out/%08d' % image.frame)

    @staticmethod
    def _parse_lidar_data( weak_ref, image):
        self = weak_ref()
        if not self:
            return

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        lidar_data = np.array(points[:, :2])
        dim = (self._camera_man_args['agent_image_width'],
            self._camera_man_args['agent_image_height'])
        lidar_data *= min(dim) / 100.0
        lidar_data += (0.5 * dim[0], 0.5 * dim[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (dim[0], dim[1], 3)
        lidar_img = np.zeros(lidar_img_size)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        # Loading lidar data to the corresponding spot
        self._current_lidar_observation = lidar_img

        if self._render_mode == 'agent_lidar':
            self.surface = pygame.surfarray.make_surface(lidar_img)

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
class World(object):
    def __init__(self, carla_world, hud, world_args):

        # Args related, clean later
        # self.actor_role_name = args.rolename
        # self._actor_filter = args.filter
        self._gamma = 1.0 # args.gamma

        # Dictopnary for world customization
        self._world_args = world_args
        self._render_mode = world_args['render_mode']

        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0

        # Processing world_args
        self._vehicle_bp = world_args['vehicle_bp']
        self._agent_spwan_point = world_args['agent_spawn_point']
        self._world_args = world_args # Required to configure the camera manager down the line
        self._camera_man_args = world_args[ 'camera_man_args']

        self._seed = world_args['seed']
        self.set_seed()

        # self.world.set_timeout( 2.0)
        self.restart()

        if self.hud is not None:
            self.world.on_tick(self.hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def set_seed( self, seed = None):
        if seed is not None:
            self._seed = seed

        random.seed( self._seed)
        np.random.seed( self._seed)

    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Ready to fetch blueprints
        blueprint_library = self.world.get_blueprint_library()

        # Create agent's vehicle
        vehicles_bps = blueprint_library.filter('vehicle')
        agent_vehicle_bp = vehicles_bps.filter( self._vehicle_bp)[0]
        agent_vehicle_bp.set_attribute('role_name', 'agent')

        # TODO: Color setting: not much use for now
        # if blueprint.has_attribute('color'):
        #     color = random.choice(blueprint.get_attribute('color').recommended_values)
        #     blueprint.set_attribute('color', color)

        # TODO: What is this and do we need it ?
        # if blueprint.has_attribute('driver_id'):
        #     driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
        #     blueprint.set_attribute('driver_id', driver_id)

        # TODO: Proabably useless for our experi,emt
        # if blueprint.has_attribute('is_invincible'):
        #     blueprint.set_attribute('is_invincible', 'true')
        # Spawn the player.

        if self.player is not None:
            self.destroy()
            # Using default spawn point
            # TODO: Better parametrization / randomization of the spawn poinjt
            self.player = self.world.try_spawn_actor(agent_vehicle_bp, self._agent_spwan_point)

        while self.player is None:
            self.player = self.world.try_spawn_actor(agent_vehicle_bp, self._agent_spwan_point)

        # Custom: Attaching RGB front camera to agent
        # Set up the sensors.
        # TODO: Remove useless sensors for RL
        self.collision_sensor = CollisionSensor(self.player, self.hud)

        if self._render_mode == 'human':
            self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
            self.gnss_sensor = GnssSensor(self.player)

        # camera manager config
        self.camera_manager = CameraManager(self.player, self.hud, self._camera_man_args)
        # Skipped default sensor creation
        # self.camera_manager.transform_index = cam_pos_index
        # self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        # Do not render if hud is not created: this probably means either we are not rendering or, rendering only the agent's view
        if self.hud is not None:
            self.hud.notification(actor_type)

        # TODO: Parametrize weather later
        self.world.set_weather( carla.WeatherParameters.Default)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        if self.hud is not None:
            self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        if self.hud is not None:
            self.hud.render(display)

    def destroy_sensors(self):
        if hasattr( self.camera_manager, '_display_cam_sensor'):
            # print( "# DEBUG: Destroing diplay cam sensor")
            if self.camera_manager._display_cam_sensor is not None:
                self.camera_manager._display_cam_sensor.destroy()

        if hasattr( self.camera_manager, '_agent_front_cam_sensor'):
            # print( "# DEBUG: Destroing agent cam sensor")
            if self.camera_manager._agent_front_cam_sensor is not None:
                self.camera_manager._agent_front_cam_sensor.destroy()

        if hasattr( self.camera_manager, '_agent_lidar_sensor'):
            # print( "# DEBUG: Destroing agent lidar sensor")
            if self.camera_manager._agent_lidar_sensor is not None:
                self.camera_manager._agent_lidar_sensor.destroy()

        # TODO: Remove the fopllowing lines if the corresponding sensor is not used
        # Namely LaneInvasion and GNSS
        if self.collision_sensor is not None:
            self.collision_sensor.sensor.destroy()

        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.sensor.destroy()

        if self.gnss_sensor is not None:
            self.gnss_sensor.sensor.destroy()

        if self.player is not None:
            self.player.destroy()

        self.camera_manager.sensor = None
        self.camera_manager.index = None
        self.player = None

    def destroy(self):
        self.destroy_sensors()

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================
class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================
class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        # self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((250, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        # TODO: Remove or tweak help message, probably not need though
        # self.help.render(display)
