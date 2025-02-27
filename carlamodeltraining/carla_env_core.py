import sys
# Add the 'carla' folder (which contains the agents package) to sys.path.
sys.path.insert(0, "C:/Program Files/Carla/WindowsNoEditor/PythonAPI/carla")
print("sys.path:", sys.path)

import carla
import numpy as np
import random
import time
from agents.navigation.basic_agent import BasicAgent  # Now 'agents' is top-level

class CarlaLaneKeepingEnv:
    def __init__(self, num_pedestrians=10):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Cleanup any previous vehicles
        self.cleanup_previous_vehicles()

        # Spawn a vehicle
        self.vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        self.vehicle = None
        for point in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(self.vehicle_bp, point)
                self.spawn_point = point  # Save the first successful spawn point
                print(f"✅ Vehicle spawned at: {point}")
                break
            except RuntimeError:
                continue
        if not self.vehicle:
            raise RuntimeError("❌ No valid spawn point found!")

        # Attach sensors
        self.attach_spectator_to_vehicle()
        self.lidar_sensor = self.attach_lidar_sensor()

        # Spawn pedestrians
        self.spawn_pedestrians(num_pedestrians)

        # Environment properties
        self.lane_center = self.spawn_point.location.y
        self.done = False

        # Extended episode parameters
        self.step_count = 0
        self.max_steps = 900  # e.g. 900 steps ~90 sec if each step ~0.1 sec

        # Instantiate the BasicAgent for map-based guidance.
        # Set a target speed (in Km/h) and choose a random destination.
        self.basic_agent = BasicAgent(self.vehicle, target_speed=20)
        destination = random.choice(spawn_points).location
        self.basic_agent.set_destination(destination)
        print("🔄 BasicAgent instantiated for map guidance.")

    def reset(self):
        """Reset the environment and vehicle position."""
        self.vehicle.set_transform(self.spawn_point)
        self.vehicle.apply_control(carla.VehicleControl(steer=0.0, throttle=0.5, brake=0.0))
        self.done = False
        self.step_count = 0  # Reset step counter
        self.attach_spectator_to_vehicle()
        self.closest_obstacle_distance = 50.0  # Default max distance
        state = self.get_state()
        return state

    def get_state(self):
        # Return raw (unnormalized) state values.
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        lateral_position = transform.location.y - self.lane_center
        heading_angle = transform.rotation.yaw
        speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2)
        obstacle_distance = self.closest_obstacle_distance
        return np.array([lateral_position, heading_angle, speed, obstacle_distance], dtype=np.float32)

    def cleanup_previous_vehicles(self):
        """Remove all existing vehicles."""
        actors = self.world.get_actors().filter("vehicle.*")
        for actor in actors:
            actor.destroy()
        print("✅ Cleaned up previous vehicles.")

    def attach_spectator_to_vehicle(self):
        """Attach the spectator camera behind the vehicle."""
        spectator = self.world.get_spectator()
        transform = self.vehicle.get_transform()
        offset_distance = -10.0  # 10 meters behind
        offset_height = 5.0      # 5 meters above
        yaw_radians = np.radians(transform.rotation.yaw)
        camera_x = transform.location.x + offset_distance * np.cos(yaw_radians)
        camera_y = transform.location.y + offset_distance * np.sin(yaw_radians)
        camera_z = transform.location.z + offset_height
        spectator_transform = carla.Transform(
            carla.Location(x=camera_x, y=camera_y, z=camera_z),
            carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw)
        )
        spectator.set_transform(spectator_transform)
        print(f"📷 Camera attached at: {camera_x}, {camera_y}, {camera_z}")

    def attach_lidar_sensor(self):
        """Attach a LiDAR sensor."""
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('rotation_frequency', '10')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
        lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        lidar_sensor.listen(lambda data: self.process_lidar_data(data))
        print("📡 LiDAR sensor attached.")
        return lidar_sensor

    def process_lidar_data(self, data):
        """Process LiDAR data."""
        points = np.array([[detection.point.x, detection.point.y, detection.point.z] for detection in data])
        filtered_points = np.empty((0, 3))
        if len(points) > 0:
            filtered_points = points[points[:, 2] > 0.5]
            if len(filtered_points) > 0:
                min_distance = np.min(np.sqrt(filtered_points[:, 0] ** 2 + filtered_points[:, 1] ** 2))
            else:
                min_distance = 50.0
        else:
            min_distance = 50.0
        self.closest_obstacle_distance = min_distance
        print(f"📡 LiDAR detected {len(filtered_points)} points. Closest obstacle: {min_distance:.2f}m")

    def spawn_pedestrians(self, num_pedestrians=10):
        """Spawn pedestrians."""
        blueprint_library = self.world.get_blueprint_library()
        walker_bp = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")
        all_spawn_points = self.world.get_map().get_spawn_points()
        sidewalk_spawn_points = [p for p in all_spawn_points if abs(p.location.y) > 3]
        if len(sidewalk_spawn_points) < num_pedestrians:
            print("⚠️ Not enough pedestrian-safe spawn points. Reducing count.")
            num_pedestrians = len(sidewalk_spawn_points)
        pedestrian_actors = []
        controllers = []
        for i in range(num_pedestrians):
            walker = random.choice(walker_bp)
            spawn_point = sidewalk_spawn_points[i]
            nearby_pedestrians = self.world.get_actors().filter("walker.pedestrian.*")
            occupied = any(actor.get_location().distance(spawn_point.location) < 2.0 for actor in nearby_pedestrians)
            if occupied:
                print(f"🚨 Spawn position occupied! Skipping pedestrian {i}.")
                continue
            walker_actor = self.world.try_spawn_actor(walker, spawn_point)
            if walker_actor:
                pedestrian_actors.append(walker_actor)
                controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker_actor)
                if controller:
                    controller.start()
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                    controller.set_max_speed(random.uniform(0.5, 1.5))
                    controllers.append(controller)
        print(f"🚶 Successfully spawned {len(pedestrian_actors)} pedestrians with AI movement.")

    def step(self, action):
        # Increment step counter
        self.step_count += 1
        # Get current state and extract values
        state = self.get_state()
        lateral_position, heading_angle, speed, obstacle_distance = state

        # Get the BasicAgent's recommended control for map guidance.
        basic_control = self.basic_agent.run_step()
        desired_steer = basic_control.steer  # desired steering from map-based planning

        # Initialize reward
        reward = 0.0
        # Termination conditions: allow up to 3.0m lateral deviation, etc.
        if abs(lateral_position) > 3.0:
            self.done = True
            reward = -100.0
        elif obstacle_distance < 5.0:
            self.done = True
            reward = -100.0
        elif self.step_count >= self.max_steps:
            self.done = True
            reward = 0.0
        else:
            speed_bonus = speed * 0.6
            lane_penalty = (abs(lateral_position) ** 2) * 1.5
            heading_penalty = abs(heading_angle) / 20.0
            reward = speed_bonus - lane_penalty - heading_penalty
            # Add a penalty if RL steering deviates from the BasicAgent's guidance.
            reward -= abs(desired_steer) * 2.0

        max_steer = 0.35
        # Compute the RL agent's steering based on the discrete action
        if action == 0:
            agent_steer = -max_steer
        elif action == 1:
            agent_steer = 0.0
        elif action == 2:
            agent_steer = max_steer
        else:
            agent_steer = 0.0

        # Blended steering: if off-center, apply a corrective term blended with agent's action.
        if abs(lateral_position) > 0.3:
            corrective_steer = -max_steer * (lateral_position / 3.0)
            blend_factor = min(1.0, abs(lateral_position) / 0.5)
            steer_direction = (1 - blend_factor) * agent_steer + blend_factor * corrective_steer
        else:
            steer_direction = agent_steer

        # Optionally, blend in a fraction of the BasicAgent's desired steer.
        guidance_blend = 0.3  # 30% weight to BasicAgent's recommendation
        steer_direction = (1 - guidance_blend) * steer_direction + guidance_blend * desired_steer

        throttle = 0.5
        control = carla.VehicleControl(throttle=throttle, steer=steer_direction)
        self.vehicle.apply_control(control)
        self.world.tick()
        self.attach_spectator_to_vehicle()

        # Debug information:
        velocity = self.vehicle.get_velocity()
        current_speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        print(f"Step: {self.step_count} | Control: throttle={throttle}, steer={steer_direction:.2f} | "
              f"Lateral offset: {lateral_position:.2f}, Heading: {heading_angle:.2f}, "
              f"Speed: {current_speed:.2f} m/s, Reward: {reward:.2f}, Desired steer: {desired_steer:.2f}")

        return state, float(reward), self.done

    def close(self):
        if self.lidar_sensor is not None:
            self.lidar_sensor.destroy()
        if self.vehicle is not None:
            self.vehicle.destroy()
        print("✅ Environment closed and actors cleaned up.")
