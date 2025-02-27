import carla

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Get all actors in the environment
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')  # Filter only vehicles

    # Destroy all vehicles
    for vehicle in vehicles:
        try:
            vehicle.destroy()
            print(f"Destroyed vehicle: {vehicle.id}")
        except RuntimeError as e:
            print(f"Error destroying vehicle {vehicle.id}: {e}")

if __name__ == "__main__":
    main()
