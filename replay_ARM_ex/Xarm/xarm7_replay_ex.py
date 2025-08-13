import argparse
import os
import sys
import time
import h5py
import numpy as np
from datetime import date, datetime
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import pyrealsense2 as rs

# 将 robotics tool 相关包添加到路径
sys.path.append('/home/onestar/hny/vla/lerobot/BestMan_Xarm/Robotics_API')
# from Bestman_real_xarm6 import *
from Bestman_Real_Xarm7 import *

def calculate_new_pose(x, y, z, quaternion, distance):
    """
    基于给定的6D位姿 (x, y, z, 四元数), 计算沿着 z 轴“负方向”平移 distance 后的新位姿。
    """
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    z_axis = rotation_matrix[:, 2]        # 取出姿态矩阵的 z 轴 (第三列)
    new_position = np.array([x, y, z]) - distance * z_axis
    return new_position[0], new_position[1], new_position[2], quaternion


if __name__ == "__main__":
    # 初始化 XArm
    bestman = Bestman_Real_Xarm7('192.168.1.224', None, None)

    # 准备数据
    input_file = '/home/onestar/tonglu/data_replay/episode_0.hdf5'
    with h5py.File(input_file, 'r') as f_in:
        # 读取位置与姿态
        xyz_data = f_in['action'][:, :3]
        q_data = f_in['action'][:, 3:7]
        gripper_data = f_in['action'][:, 7]

    # 将四元数转换为欧拉角（XYZ顺序，单位：度）
    rotation = R.from_quat(q_data)
    euler_angles_data = rotation.as_euler('xyz', degrees=True)

    # 主循环，依次执行轨迹中的动作
    for i in range(xyz_data.shape[0]):
        xyz_action = xyz_data[i]           # [x, y, z]
        euler_action = euler_angles_data[i]  # [roll, pitch, yaw] in degrees
        gripper_raw = gripper_data[i]      # 原始 gripper 值 (0~1)

        # 计算 gripper 的整数值 (0~255)，并保持与原逻辑一致
        gripper_cmd = int((1 - gripper_raw) * 255)

        # 计算与 gripper 线性关联的位移距离
        # 当 gripper_cmd=0 -> distance=0.09; 当 gripper_cmd=255 -> distance=0.105
        current_distance = 0.082 + 0.015 * (gripper_cmd / 255.0)
        # current_distance = 0.105

        # 计算新的平移后坐标
        quat_action = q_data[i]
        x_new, y_new, z_new, _ = calculate_new_pose(
            xyz_action[0], xyz_action[1], xyz_action[2],
            quat_action, current_distance
        )
        print("x_new, y_new, z_new", x_new, y_new, z_new)
        # 设置机器人状态 (注意：此方法根据你的机器人接口可自行调整)
        bestman.robot.set_state(0)

        # 将目标位姿换算到 mm 和度
        bestman.robot.set_position(
            x_new * 1000,   # x (mm)
            y_new * 1000,   # y (mm)
            z_new * 1000,   # z (mm)
            euler_action[0],  # roll (deg)
            euler_action[1],  # pitch (deg)
            euler_action[2]   # yaw (deg)
        )

        # 控制抓手
        # bestman.gripper_goto(gripper_cmd)
        bestman.gripper_goto_robotiq(gripper_cmd)

        # 可视需要在此插入合适的延时
        time.sleep(1/10)
