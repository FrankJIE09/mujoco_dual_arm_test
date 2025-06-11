import numpy as np

def rpy_to_quaternion(roll, pitch, yaw):
    """
    将 RPY 欧拉角 (Tait-Bryan ZYX) 转换为四元数。

    Args:
        roll (float): 绕 X 轴的旋转角度 (弧度)
        pitch (float): 绕 Y 轴的旋转角度 (弧度)
        yaw (float): 绕 Z 轴的旋转角度 (弧度)

    Returns:
        numpy.ndarray: 对应的四元数 [w, x, y, z]
    """
    # 计算每个角度的一半
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # 按 ZYX 顺序计算四元数
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])

# --- 使用示例 ---
if __name__ == '__main__':
    # 定义 RPY 角度 (单位：度)
    roll_deg =90
    pitch_deg = 0
    yaw_deg = 0

    # 将角度转换为弧度
    roll_rad = np.radians(roll_deg)
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)

    # 调用函数进行转换
    quaternion = rpy_to_quaternion(roll_rad, pitch_rad, yaw_rad)

    print(f"RPY 角度 (度): Roll={roll_deg}, Pitch={pitch_deg}, Yaw={yaw_deg}")
    print(f"转换后的四元数 (w, x, y, z): {quaternion}")

    # 验证: 四元数的模长应为 1
    norm = np.linalg.norm(quaternion)
    print(f"四元数的模长: {norm:.6f}") # 应该非常接近 1.0