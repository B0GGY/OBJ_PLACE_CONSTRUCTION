import numpy as np
import quaternion

def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_rel_pose_change(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2

    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1

    return dx, dy, do  # was do

def quaternion_to_euler_rad(quaternion):
    """
    将四元数转换为欧拉角（以弧度表示）
    输入: quaternion -> [w, x, y, z]
    输出: 欧拉角 -> [roll, pitch, yaw] (弧度)
    """
    w, x, y, z = quaternion

    # 计算 Roll (绕 X 轴)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    # 计算 Yaw (绕 Z 轴)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    # 计算 Pitch (绕 Y 轴)
    sin_pitch = 2 * (w * y - z * x)
    sin_pitch = np.clip(sin_pitch, -1.0, 1.0)  # 防止数值误差导致超出范围
    pitch = np.arcsin(sin_pitch)
    print([roll, pitch, yaw])
    if abs(abs(roll)-3.14) <0.01 and abs(abs(yaw)-3.14) <0.01:
        pitch = -np.pi-pitch
        roll = 0
        yaw = 0
    if pitch > 0 :
        pitch = -2*np.pi + pitch
    return np.array([roll, pitch, yaw])


def get_new_pose(pose, rel_pose_change):
    x, y, o = pose
    dx, dy, do = rel_pose_change

    global_dx = dx * np.sin(np.deg2rad(o)) + dy * np.cos(np.deg2rad(o))
    global_dy = dx * np.cos(np.deg2rad(o)) - dy * np.sin(np.deg2rad(o))
    x += global_dy
    y += global_dx
    o += np.rad2deg(do)
    if o > 180.:
        o -= 360.

    return x, y, o


def threshold_poses(coords, shape):
    coords[0] = min(max(0, coords[0]), shape[0] - 1)
    coords[1] = min(max(0, coords[1]), shape[1] - 1)
    return coords
