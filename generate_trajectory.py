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
    #{"position": np.array([0.5, 0.2, 0.7]), "radius": 0.05}
]

# Generate the XML file for the scene
model_path = generate_scene_xml(start, end, obstacles, "obstacles.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Launch the Viewer
v = viewer.launch_passive(model, data)
time.sleep(5)  # Wait for the viewer to initialize

# Artificial potential field parameters
k_att = 60.0  # Attractive force gain
k_rep = 1.0  # Initial repulsive force gain
d0_multiplier = 4  # Multiplier for the repulsive force range
tolerance = 0.01  # Termination condition
k_offset = 0.01  # Offset gain

# Record the trajectory
trajectory = [start.copy()]
print("Start:", start)

# Function to compute the artificial potential field forces (attractive and repulsive)
def compute_apf_force(position, k_rep):
    # Attractive force
    F_att = -k_att * (position - end)
    print("F_att:", F_att)
    # Repulsive force
    F_rep_total = np.zeros_like(position)  # Initialize total repulsive force
    #if no obstacles, return only the attractive force
    if len(obstacles) == 0:
        return F_att, 0
    for obstacle in obstacles:
        dist_to_obstacle = np.linalg.norm(position - obstacle["position"])
        d0 = d0_multiplier * obstacle["radius"]
        if dist_to_obstacle < d0:
            F_rep = k_rep * (1.0 / dist_to_obstacle - 1.0 / d0) * \
                    (1.0 / dist_to_obstacle**2) * (position - obstacle["position"]) / dist_to_obstacle
            
            F_rep_total += F_rep  # Accumulate repulsive forces from all obstacles
            
            # Compute dynamic perpendicular offset
            if np.linalg.norm(F_rep_total) > 1e-6 and np.linalg.norm(F_att) > 1e-6:
                # Attempt cross-product of F_rep_total and F_att
                offset = np.cross(F_rep_total, F_att)
                if np.linalg.norm(offset) < 1e-6:  # If result is near zero, use a random reference vector
                    random_vector = np.random.uniform(-1, 1, size=3)  # Generate random 3D vector
                    random_vector /= np.linalg.norm(random_vector) + 1e-6  # Normalize
                    offset = np.cross(F_rep_total, random_vector)
            else:
                # Use random vector if both forces are nearly zero
                random_vector = np.random.uniform(-1, 1, size=3)  # Generate random 3D vector
                random_vector /= np.linalg.norm(random_vector) + 1e-6  # Normalize
                offset = np.cross(F_rep_total, random_vector)

            # Normalize and scale the offset
            offset = k_offset * offset / (np.linalg.norm(offset) + 1e-6)

            # Apply offset to the total repulsive force
            F_rep_total += offset
            
        else:
            F_rep = np.zeros_like(position)

    # Return the total force and the minimum distance to the nearest obstacle
    return F_att + F_rep_total, np.min([np.linalg.norm(position - obs["position"]) for obs in obstacles])

# Generate trajectory with obstacle avoidance
current_position = start
while np.linalg.norm(current_position - end) > tolerance:  # Termination condition
    # Compute total force and the minimum distance to obstacles
    force, min_dist_to_obstacle = compute_apf_force(current_position, k_rep)

    # If the repulsive force is applied (distance to obstacle < d0), reduce k_rep
    #if min_dist_to_obstacle < d0_multiplier * min([obs["radius"] for obs in obstacles]):
        #k_rep *= 0.99  # Reduce k_rep by 80% for each iteration

    # Update position
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
print("Trajectory saved to trajectory.csv")

time.sleep(5)  # Wait for 5 seconds
v.close()  # Close the viewer