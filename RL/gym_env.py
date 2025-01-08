import gym
from gym import spaces
import carla
import numpy as np
import cv2
from stable_baselines3.common.env_util import make_vec_env




class CarlaEnv(gym.Env):
    """Custom Gym Environment for Carla."""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CarlaEnv, self).__init__()
        # Connect to Carla
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # Set up blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Example: 0 = Left, 1 = Forward, 2 = Right
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 160, 3), dtype=np.uint8)

        # Vehicle and camera attributes
        self.vehicle = None
        self.camera = None
        self.camera_data = None

    def reset(self):
        # Destroy previous actors
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.camera is not None:
            self.camera.destroy()

        # Spawn vehicle
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        # Attach camera
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '160')
        camera_bp.set_attribute('image_size_y', '80')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

        # Set up camera listener
        self.camera.listen(lambda image: self._process_camera(image))

        # Wait for the first camera frame
        while self.camera_data is None:
            pass

        return self.camera_data

    def step(self, action):
        # Apply action
        if action == 0:  # Turn Left
            self.vehicle.apply_control(carla.VehicleControl(steer=-0.5, throttle=0.5))
        elif action == 1:  # Go Forward
            self.vehicle.apply_control(carla.VehicleControl(steer=0.0, throttle=0.5))
        elif action == 2:  # Turn Right
            self.vehicle.apply_control(carla.VehicleControl(steer=0.5, throttle=0.5))

        # Wait for next frame
        self.world.tick()

        # Compute reward
        reward = self._compute_reward()
        done = self._check_done()

        return self.camera_data, reward, done, {}

    def close(self):
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.camera is not None:
            self.camera.destroy()

    def _process_camera(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        rgb_image = array[:, :, :3]
        self.camera_data = cv2.resize(rgb_image, (160, 80))

    def _compute_reward(self):
        # Placeholder reward function
        return 1.0  # Reward for staying alive

    def _check_done(self):
        # Placeholder done condition
        return False


env = make_vec_env(CarlaEnv, n_envs=1)  # Single environment for now
    