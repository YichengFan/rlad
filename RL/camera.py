import carla
import numpy as np
import cv2

def main():
    # Connect to Carla Simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    try:
        # Get the blueprint library
        blueprint_library = world.get_blueprint_library()

        # Spawn a vehicle
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        # Add a front-facing camera sensor
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')  # Width
        camera_bp.set_attribute('image_size_y', '480')  # Height
        camera_bp.set_attribute('fov', '90')           # Field of View

        # Attach the camera to the vehicle
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Adjust position as needed
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Camera data callback
        def process_image(image):
            # Convert raw image to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # RGBA format
            rgb_image = array[:, :, :3]  # Drop alpha channel
            cv2.imshow('Camera Feed', rgb_image)
            cv2.waitKey(1)

        # Listen to the camera stream
        camera.listen(process_image)

        # Run simulation for a while
        import time
        time.sleep(30)  # Keep the simulation running for 30 seconds

    finally:
        print("Destroying actors...")
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == '__main__':
    main()
