# FastUMI_replay_singleARM
This is the demo for FastUMI data replay.

# How to download demo:
We use **unplug_charger** dataset as examples.

```bash
# need to login huggingface
hf auth login --token <YOUR_TOKEN>
#use hf-mirror
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download IPEC-COMMUNITY/FastUMI-Data unplug_charger.tar.gz.part-001 --local-dir ~/fast_umi/
huggingface-cli download --repo-type dataset --resume-download IPEC-COMMUNITY/FastUMI-Data unplug_charger.tar.gz.part-002 --local-dir ~/fast_umi/
huggingface-cli download --repo-type dataset --resume-download IPEC-COMMUNITY/FastUMI-Data unplug_charger.tar.gz.part-003 --local-dir ~/fast_umi/
```

# Merging Data:
```bash
cd <your data>
cat unplug_charger.tar.gz.part-00* > unplug_charger.tar.gz
```

# Extract the Dataset:
```bash
tar -xzvf unplug_charger.tar.gz
```
# Data Process:
## Step 1:
Merge all episodes(unplug_charger_v0 - unplug_charger_v9)
```bash
python3 merge_episodes.py 
```
## Step 2:
Convert the FastUMI data from the camera coordinate system to the base coordinate system, and normalize the gripper width.
This will consume a huge memory.

### Configure Local Robot Parameters

Before running the program, update the **base position** and **orientation parameters** according to your robot's loacl setup:

```python
#Ex:
# Robot base position (in meters)
base_x, base_y, base_z = 0.51043, -0.00588, 0.35984
# Robot base orientation (in degrees, converted to radians)
base_roll, base_pitch, base_yaw = np.deg2rad([179.94725, -89.999981, 0.0])
```
Then run the code:
```bash
conda env create -f environment.yml
conda activate FastUMI
python3 coordinate_transform.py
```
After this configuration, the system will provide the pose based on the **local robot base** in the following format:

[Pos X, Pos Y, Pos Z, Q_X, Q_Y, Q_Z, Q_W, gripper_open_width]

Where:
- **Pos X, Pos Y, Pos Z**: Position of the robot’s end-effector relative to the local robot base (in meters).
- **Q_X, Q_Y, Q_Z, Q_W**: Orientation of the end-effector in quaternion form.
- **gripper_open_width**: Current opening width of the gripper.

# How to replay the data on your own robot:
Here is a generic logic example:
```python
import time
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------- Generic Robot Class (to be implemented by the user) ----------------
class Robot:
    def __init__(self, ip):
        """
        Initialize the robot connection.
        ip: IP address of the robot controller
        """
        # TODO: Initialize your robot's SDK/API here
        pass

    def set_state(self, state: int):
        """
        Set the robot's state.
        Example: 0 = ready/motion enabled, 1 = idle, etc.
        """
        pass

    def move_pose(self, x_m, y_m, z_m, roll_deg, pitch_deg, yaw_deg):
        """
        Move to a Cartesian pose.
        Units:
            - Position in meters
            - Orientation in degrees (roll, pitch, yaw)
        """
        pass

    def gripper_goto(self, cmd_0_255: int):
        """
        Move the gripper to the target position.
        Range:
            - 0 = fully open (or closed depending on your convention)
            - 255 = fully closed (or open)
        """
        pass


# ---------------- Utility Function ----------------
def calculate_new_pose(x, y, z, quaternion, distance):
    """
    Given a position (x, y, z) and an orientation (quaternion),
    compute a new position shifted along the negative local Z-axis
    by 'distance' meters.
    """
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    z_axis = rotation_matrix[:, 2]
    new_position = np.array([x, y, z]) - distance * z_axis
    return new_position[0], new_position[1], new_position[2], quaternion


# ---------------- Main Script ----------------
if __name__ == "__main__":
    # 1. Initialize the robot (replace with your own implementation)
    robot = Robot(ip="192.168.1.100")  # Replace with actual IP

    # 2. Load trajectory data from HDF5 file
    input_file = "/path/to/episode_0.hdf5"
    with h5py.File(input_file, "r") as f:
        xyz_data = f["action"][:, :3]
        q_data = f["action"][:, 3:7]
        gripper_data = f["action"][:, 7]

    # Convert quaternions to Euler angles (XYZ order, degrees)
    euler_angles_data = R.from_quat(q_data).as_euler("xyz", degrees=True)

    # 3. Execute the trajectory
    for i in range(len(xyz_data)):
        xyz_action = xyz_data[i]
        euler_action = euler_angles_data[i]
        gripper_raw = gripper_data[i]

        # Map gripper value from 0~1 to integer 0~255
        gripper_cmd = int((1 - gripper_raw) * 255)

        # Compute distance offset based on gripper position
        current_distance = 0.082 + 0.015 * (gripper_cmd / 255.0)

        # Apply position offset along negative Z-axis
        quat_action = q_data[i]
        x_new, y_new, z_new, _ = calculate_new_pose(
            xyz_action[0], xyz_action[1], xyz_action[2],
            quat_action, current_distance
        )

        # Set robot state
        robot.set_state(0)

        # Move to the target pose
        robot.move_pose(
            x_new, y_new, z_new,
            euler_action[0], euler_action[1], euler_action[2]
        )

        # Control gripper
        robot.gripper_goto(gripper_cmd)

        # Small delay
        time.sleep(0.1)
```
