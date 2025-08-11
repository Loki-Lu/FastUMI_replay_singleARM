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
