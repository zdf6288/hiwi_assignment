import numpy as np
import mujoco
import mujoco.viewer as viewer
import time

# Control parameters tuning
k = 10.0  # Position control gain
d = 5  # Null-space projection gain
tolerance = 0.02  # Position tolerance to reach the target

# Obstacle avoidance parameters (set to 0 to disable)
alpha= 0.15  # Weight for joint-based avoidance
beta = 0.2  # Weight for joint midpoint-based avoidance

# Load panda model with obstacles
model_path = "panda.xml"  # Path to the Panda model
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Launch Viewer
v = viewer.launch_passive(model, data)

# Get the body ID and initial position of the hand
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
if hand_id == -1:
    print("Body 'hand' not found in the model.")
    exit()

# Get the initial position of the hand
initial_position = data.xpos[hand_id]  # [x, y, z]
print(f"Hand Initial Position: {initial_position}")

# Load the desired trajectory from a CSV file
trajectory = np.loadtxt("trajectory.csv", delimiter=",", skiprows=1)
print("Generated Trajectory:")
print(trajectory)

# Sample 20 intermediate points between the initial position and the first trajectory point
# Later, we will move the hand to these intermediate points before following the trajectory
first_trajectory_point = trajectory[0]
intermediate_points = np.linspace(initial_position, first_trajectory_point, 20)

# Get joint limits from the model
q_min = model.jnt_range[:, 0]
q_max = model.jnt_range[:, 1]
q_center = 0.5 * (q_min + q_max)

# Get joint IDs
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}") for i in range(7)]  # Assuming Panda has 7 joints

def is_obstacle(geom_id):
    # name of geom_id
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    # check if the geom_id is an obstacle
    return geom_name is not None and "obstacle" in geom_name

# Define function to move to a target position
def move_to_target(target_position, k, d, alpha = 0.5, beta = 0.5, min_distance=0.1):
    current_pos = data.xpos[hand_id]
    while np.linalg.norm(target_position - current_pos) > tolerance:
        
        # Compute task-space error
        error = target_position - current_pos

        # Compute the Jacobian matrix
        J = np.zeros((3, model.nv))  # Task-space (x, y, z) vs. joint-space
        mujoco.mj_jac(model, data, J, None, current_pos, hand_id)

        # Compute the pseudo-inverse of the Jacobian
        J_pseudo = np.linalg.pinv(J)

        # Null-space projection
        I = np.eye(model.nv)
        N = I - J_pseudo @ J

        # Initialize gradients
        joint_gradient = np.zeros((model.nv, 3))  # Gradient for joints
        midpoint_gradient = np.zeros((model.nv, 3))  # Gradient for joint midpoints
        
        # Compute obstacle avoidance gradient for all obstacles
        for geom_id in range(model.ngeom):
            if is_obstacle(geom_id):  # Check if this geom is an obstacle
                obstacle_pos = data.geom_xpos[geom_id]  # Obstacle position (x, y, z)
                
                # (1) Joint-based avoidance
                for joint_id in joint_ids:
                    joint_pos = data.xpos[joint_id]  # Joint position in Cartesian space
                    distance_vector = joint_pos - obstacle_pos
                    distance = np.linalg.norm(distance_vector)

                    if distance > min_distance:  # Avoid division by zero
                        weight = 1 / (distance**2)
                        joint_gradient[joint_id, :] += weight * (distance_vector / distance)
                    else:
                        joint_gradient[joint_id, :] += 10.0 * (distance_vector / max(distance, 1e-6))

                # (2) Joint midpoint-based avoidance
                for i in range(len(joint_ids) - 1):  # For each pair of adjacent joints
                    joint_pos_1 = data.xpos[joint_ids[i]]  # Position of joint i
                    joint_pos_2 = data.xpos[joint_ids[i+1]]  # Position of joint i+1
                    
                    # Calculate the midpoint between joint i and joint i+1
                    midpoint = 0.5 * (joint_pos_1 + joint_pos_2)
                    
                    # Calculate distance vector and distance to the obstacle
                    distance_vector = midpoint - obstacle_pos
                    distance = np.linalg.norm(distance_vector)

                    if distance > min_distance:  # Avoid division by zero
                        weight = 1 / (distance**2)
                        midpoint_gradient[:, :] += weight * (distance_vector / distance)
                    else:
                        midpoint_gradient[:, :] += 10.0 * (distance_vector / max(distance, 1e-6))

        # Combine gradients
        joint_gradient_joint_space = np.sum(joint_gradient, axis=1)  # Sum x, y, z contributions for each joint
        midpoint_gradient_joint_space = np.sum(midpoint_gradient, axis=1)  # Sum x, y, z contributions for midpoints
        
        # Total gradient
        total_gradient = alpha * joint_gradient_joint_space + beta * midpoint_gradient_joint_space
        
        # Compute joint velocities
        q_dot = J_pseudo @ (k * error) + N @ (d * (q_center - data.qpos) + total_gradient)

        # Update joint velocities
        data.qvel[:] = q_dot

        # Simulate and render
        time.sleep(0.02)  # Reduce the simulation speed for visualization
        
        # Perform simulation step
        if v.is_running():
            time.sleep(0.02)    # Reduce the simulation speed for visualization
            mujoco.mj_step(model, data)
            v.sync()
        
        # Update the current position of the hand
        current_pos = data.xpos[hand_id]

# Move to the first trajectory point using intermediate points
time.sleep(2)  # Wait for 2 seconds
for point in intermediate_points:
    move_to_target(point, k, d, alpha, beta)

# Follow the entire trajectory
for target_pos in trajectory:
    move_to_target(target_pos, k, d, alpha, beta)

print("Trajectory tracking completed!")
time.sleep(3)  # Wait for 3 seconds before closing the viewer