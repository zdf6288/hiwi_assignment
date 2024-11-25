import numpy as np
import mujoco
import mujoco.viewer as viewer
import time

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

# Control parameters
k = 10.0  # Position control gain
d = 10.0  # Null-space projection gain
tolerance = 0.02  # Position tolerance to reach the target

# Get joint limits from the model
q_min = model.jnt_range[:, 0]
q_max = model.jnt_range[:, 1]
q_center = 0.5 * (q_min + q_max)

# Get joint IDs
joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i+1}") for i in range(7)]  # Assuming Panda has 7 joints

# Define function to move to a target position
def move_to_target(target_position, k, d):
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

        # Compute joint velocities
        q_dot = J_pseudo @ (k * error) + N @ (d * (q_center - data.qpos))

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
    move_to_target(point, k, d)

# Follow the entire trajectory
for target_pos in trajectory:
    move_to_target(target_pos, k, d)

print("Trajectory tracking completed!")
time.sleep(3)  # Wait for 3 seconds before closing the viewer