import numpy as np
import mujoco
import mujoco.viewer as viewer
import time
from common.generate_xml import generate_scene_xml

# set the start, end, and obstacles
start = np.array([0.3, 0, 0.7])  # Start position
end = np.array([0.7, 0, 0.7])  # End position
obstacles = [  # List of obstacles
    {"position": np.array([0.5, 0, 0.7]), "radius": 0.05},
]

# Generate the XML file for the scene
model_path = generate_scene_xml(start, end, obstacles, "obstacles.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Launch the Viewer
v = viewer.launch_passive(model, data)

# Lorentz force parameters
k_att = 30.0  # Attractive force gain
k_lorentz = 800.0  # Lorentz force gain
tolerance = 0.01  # Termination condition
k_radius = 3  # Radius of the obstacles

# Record the trajectory
trajectory = [start.copy()]
print("Start:", start)

# Function to compute Lorentz force
def compute_lorentz_force(position, last_position, decay=0.9):
    F_att = -k_att * (position - end)
    print("F_att:", F_att)
    F_lorentz_total = np.zeros_like(position)
    for obstacle in obstacles:
        dist_to_obstacle = np.linalg.norm(position - obstacle["position"])
        effective_range = k_radius * obstacle["radius"]
        if dist_to_obstacle < effective_range:
            # Compute pseudo-magnetic field
            B = (position - obstacle["position"]) / (dist_to_obstacle + 1e-6)
            B = np.cross(B, np.array([0, 1, 0]))
            B = B / (np.linalg.norm(B) + 1e-6)  # Normalize

            # Compute current vector I
            I = position - last_position

            print("I:", I)
            # Compute Lorentz force
            F_lorentz = k_lorentz * np.cross(I, B)
            print("F_lorentz:", F_lorentz)
            F_lorentz_total += F_lorentz
    return F_att + F_lorentz_total

# Generate trajectory with Lorentz force-based obstacle avoidance
current_position = start
last_position = start  # Initialize the last position as the start position

while np.linalg.norm(current_position - end) > tolerance:  # Termination condition
    # Compute total force
    force = compute_lorentz_force(current_position, last_position)
    print("Force:", force)
    # Update position
    last_position = current_position.copy()  # Update last position before moving
    current_position += 0.01 * force / np.linalg.norm(force + 1e-6)
    trajectory.append(current_position.copy())

    # Update mocap position to visualize trajectory in the viewer
    if v.is_running():
        data.mocap_pos[0] = current_position  # Update data only when Viewer is active
        # Simulation step and rendering
        time.sleep(0.1)
        mujoco.mj_step(model, data)
        v.sync()

# Print the generated trajectory
trajectory = np.array(trajectory)
print("Generated Trajectory:")
print(trajectory)

# Save the trajectory to a CSV file
np.savetxt("trajectory.csv", trajectory, delimiter=",", header="x,y,z", comments="")
print("Trajectory saved to trajectory_lorentz.csv")

v.close()