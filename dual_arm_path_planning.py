#!/usr/bin/env python3
"""
双臂机器人路径规划与仿真系统

这个模块实现了双臂机器人抓取物体的完整路径规划系统，包括：
1. 路径规划：为虚拟物体中心点规划移动轨迹
2. 路径插值：生成平滑的路径点序列
3. 逆运动学求解：为每个路径点计算双臂关节角度
4. MuJoCo仿真：可视化整个执行过程

数学原理：
- 路径规划使用多项式插值或贝塞尔曲线
- 姿态插值使用球面线性插值(SLERP)
- 约束满足：保持双臂闭环约束不变

作者: frank
日期: 2024年6月19日
"""

import numpy as np
import time
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 配置matplotlib支持中文显示
from font_config import init_chinese_font, set_custom_font

# 初始化中文字体支持
init_chinese_font(verbose=False)

# 导入双臂IK模块
from dual_arm_ik_modules.kinematic_chain import get_kinematics
from dual_arm_ik_modules.ik_solver import solve_dual_arm_ik


class DualArmPathPlanner:
    """
    双臂路径规划器类

    该类封装了双臂协同操作的路径规划功能，包括：
    - 路径点生成
    - 路径插值
    - 约束维护
    """

    def __init__(self, xml_file):
        """
        初始化路径规划器

        Args:
            xml_file (str): MuJoCo XML文件路径
        """
        self.xml_file = xml_file

        # 获取运动学模型
        print("正在初始化运动学模型...")
        self.kinematics_data = get_kinematics(xml_file)
        self.left_chain, self.right_chain, self.T_W_B1, self.T_W_B2 = self.kinematics_data

        # 从XML获取home关键帧
        self._load_home_pose()

        print("路径规划器初始化完成")

    def _load_home_pose(self):
        """从XML文件加载home关键帧位姿"""
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        home_qpos_str = root.find(".//key[@name='pos_1']").get('qpos')
        self.home_q = np.fromstring(home_qpos_str, sep=' ')

        # 计算home位姿下的虚拟物体中心位姿
        self.T_W_M_home = self._compute_virtual_object_pose(self.home_q)
        print(f"Home位姿虚拟物体中心: {self.T_W_M_home[:3, 3]}")

    def _compute_virtual_object_pose(self, q):
        """
        计算给定关节角度下的虚拟物体中心位姿

        Args:
            q (np.ndarray): 12个关节角度

        Returns:
            np.ndarray: 4x4变换矩阵，虚拟物体中心的位姿
        """
        q1 = q[:6]  # 右臂6个关节角度
        q2 = q[6:]  # 左臂6个关节角度

        # ikpy正向运动学
        fk_q1 = np.concatenate(([0], q1, [0]))
        fk_q2 = np.concatenate(([0], q2, [0]))

        T_B1_E1 = self.left_chain.forward_kinematics(fk_q1)
        T_W_E1 = self.T_W_B1 @ T_B1_E1

        # 虚拟物体中心的变换关系（与IK求解器中的定义一致）
        T_E1_M = np.eye(4)
        T_E1_M[:3, :3] = Rotation.as_matrix(Rotation.from_euler("XYZ", (0, 0, 0), degrees=True))
        T_E1_M[0, 3] = 0.0  # X轴
        T_E1_M[1, 3] = 0.0  # Y轴
        T_E1_M[2, 3] = 0.0  # Z轴

        T_W_M = T_W_E1 @ T_E1_M
        return T_W_M

    def plan_straight_line_path(self, start_pose, end_pose, num_points=20):
        """
        规划直线路径

        Args:
            start_pose (np.ndarray): 起始位姿 (4x4矩阵)
            end_pose (np.ndarray): 终点位姿 (4x4矩阵)
            num_points (int): 路径点数量

        Returns:
            list: 路径点列表，每个元素是4x4变换矩阵
        """
        print(f"规划直线路径，从 {start_pose[:3, 3]} 到 {end_pose[:3, 3]}")

        # 位置插值：线性插值
        start_pos = start_pose[:3, 3]
        end_pos = end_pose[:3, 3]
        t_values = np.linspace(0, 1, num_points)

        positions = []
        for t in t_values:
            pos = start_pos + t * (end_pos - start_pos)
            positions.append(pos)

        # 姿态插值：使用SLERP (球面线性插值)
        start_rot = Rotation.from_matrix(start_pose[:3, :3])
        end_rot = Rotation.from_matrix(end_pose[:3, :3])

        key_times = [0, 1]
        key_rotations = Rotation.concatenate([start_rot, end_rot])
        slerp = Slerp(key_times, key_rotations)

        orientations = slerp(t_values)

        # 组合位置和姿态
        path_poses = []
        for i, t in enumerate(t_values):
            pose = np.eye(4)
            pose[:3, 3] = positions[i]
            pose[:3, :3] = orientations[i].as_matrix()
            path_poses.append(pose)

        return path_poses

    def plan_circular_path(self, center_pose, radius=0.2, num_points=30, axis='z'):
        """
        规划圆形路径

        Args:
            center_pose (np.ndarray): 圆心位姿 (4x4矩阵)
            radius (float): 圆半径 (米)
            num_points (int): 路径点数量
            axis (str): 旋转轴 ('x', 'y', 或 'z')

        Returns:
            list: 路径点列表，每个元素是4x4变换矩阵
        """
        print(f"规划圆形路径，中心: {center_pose[:3, 3]}, 半径: {radius}m")

        center_pos = center_pose[:3, 3]
        center_rot = center_pose[:3, :3]

        # 生成圆形路径点
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        path_poses = []

        for angle in angles:
            # 在局部坐标系中生成圆形
            if axis == 'z':
                local_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            elif axis == 'y':
                local_pos = np.array([radius * np.cos(angle), 0, radius * np.sin(angle)])
            else:  # axis == 'x'
                local_pos = np.array([0, radius * np.cos(angle), radius * np.sin(angle)])

            # 转换到世界坐标系
            world_pos = center_pos + center_rot @ local_pos

            # 姿态保持不变（也可以根据需要添加旋转）
            pose = np.eye(4)
            pose[:3, 3] = world_pos
            pose[:3, :3] = center_rot

            path_poses.append(pose)

        return path_poses

    def plan_bezier_path(self, start_pose, end_pose, control_points=None, num_points=25):
        """
        规划贝塞尔曲线路径

        Args:
            start_pose (np.ndarray): 起始位姿 (4x4矩阵)
            end_pose (np.ndarray): 终点位姿 (4x4矩阵)
            control_points (list): 控制点列表，如果为None则自动生成
            num_points (int): 路径点数量

        Returns:
            list: 路径点列表，每个元素是4x4变换矩阵
        """
        start_pos = start_pose[:3, 3]
        end_pos = end_pose[:3, 3]

        # 如果没有提供控制点，自动生成
        if control_points is None:
            # 在起点和终点之间生成控制点，添加一些高度变化
            mid_point = (start_pos + end_pos) / 2
            mid_point[2] += 0.3  # 向上偏移30cm，形成弧形
            control_points = [mid_point]

        print(f"规划贝塞尔路径，起点: {start_pos}, 终点: {end_pos}")
        print(f"控制点: {control_points}")

        # 构建贝塞尔曲线控制点
        all_points = [start_pos] + control_points + [end_pos]
        all_points = np.array(all_points)

        # 计算贝塞尔曲线
        t_values = np.linspace(0, 1, num_points)
        positions = self._bezier_curve(all_points, t_values)

        # 姿态插值
        start_rot = Rotation.from_matrix(start_pose[:3, :3])
        end_rot = Rotation.from_matrix(end_pose[:3, :3])

        key_times = [0, 1]
        key_rotations = Rotation.concatenate([start_rot, end_rot])
        slerp = Slerp(key_times, key_rotations)
        orientations = slerp(t_values)

        # 组合位置和姿态
        path_poses = []
        for i, t in enumerate(t_values):
            pose = np.eye(4)
            pose[:3, 3] = positions[i]
            pose[:3, :3] = orientations[i].as_matrix()
            path_poses.append(pose)

        return path_poses

    def _bezier_curve(self, control_points, t_values):
        """
        计算贝塞尔曲线

        Args:
            control_points (np.ndarray): 控制点数组 (n, 3)
            t_values (np.ndarray): 参数值数组

        Returns:
            np.ndarray: 曲线点数组 (len(t_values), 3)
        """
        n = len(control_points) - 1  # 阶数
        curve_points = []

        for t in t_values:
            point = np.zeros(3)
            for i in range(n + 1):
                # 贝塞尔基函数
                bernstein = self._binomial_coefficient(n, i) * (t ** i) * ((1 - t) ** (n - i))
                point += bernstein * control_points[i]
            curve_points.append(point)

        return np.array(curve_points)

    def _binomial_coefficient(self, n, k):
        """计算二项式系数"""
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1

        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        return result

    def solve_ik_for_path(self, path_poses, initial_q=None, total_time=5.0):
        """
        为路径上的每个点求解逆运动学

        Args:
            path_poses (list): 路径点列表
            initial_q (np.ndarray): 初始关节角度，如果为None则使用home位姿
            total_time (float): 总执行时间（秒）

        Returns:
            tuple: (关节角度轨迹列表, 时间戳列表, 成功求解的点数)
        """
        if initial_q is None:
            initial_q = self.home_q.copy()

        print(f"开始为 {len(path_poses)} 个路径点求解逆运动学...")

        joint_trajectory = []
        timestamps = []
        successful_solves = 0
        current_q = initial_q.copy()

        # 生成时间戳（均匀分布）
        time_points = np.linspace(0, total_time, len(path_poses))

        for i, (target_pose, t) in enumerate(zip(path_poses, time_points)):
            print(f"求解路径点 {i + 1}/{len(path_poses)} (t={t:.2f}s)")

            # 使用上一个成功的解作为初始猜测
            T_E1_M = np.eye(4)
            T_E1_M[2, 3] = 0.0
            T_E2_M = np.eye(4)
            T_E2_M[:3, :3] = Rotation.from_euler('XYZ', (0, 0, 0), degrees=True).as_matrix()
            T_E2_M[:3, 3] = np.array([-0.8, 0, 0.0])
            solution_q, _ = solve_dual_arm_ik(target_pose, current_q, self.kinematics_data,
                                              T_E1_M=T_E1_M, T_E2_M=T_E2_M)
            if solution_q is not None:
                joint_trajectory.append(solution_q.copy())
                timestamps.append(t)
                current_q = solution_q.copy()  # 更新初始猜测
                successful_solves += 1

                # 验证解的质量
                actual_pose = self._compute_virtual_object_pose(solution_q)
                pos_error = np.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3])
                print(f"  成功! 位置误差: {pos_error:.4f}m")
            else:
                print(f"  失败! 在点 {i + 1} 处未找到解")
                # 使用上一个成功的解作为备用
                if joint_trajectory:
                    joint_trajectory.append(joint_trajectory[-1].copy())
                    timestamps.append(t)
                else:
                    joint_trajectory.append(current_q.copy())
                    timestamps.append(t)

        print(f"逆运动学求解完成: {successful_solves}/{len(path_poses)} 个点成功求解")
        return joint_trajectory, timestamps, successful_solves

    def smooth_joint_trajectory(self, joint_trajectory, timestamps=None, smoothing_factor=0.1):
        """
        对关节角度轨迹进行平滑处理

        Args:
            joint_trajectory (list): 关节角度轨迹
            timestamps (list): 时间戳列表，如果为None则使用均匀时间分布
            smoothing_factor (float): 平滑因子 (0-1)

        Returns:
            tuple: (平滑后的关节角度轨迹, 时间戳列表)
        """
        if len(joint_trajectory) <= 2:
            if timestamps is None:
                timestamps = np.linspace(0, 5.0, len(joint_trajectory))
            return joint_trajectory, timestamps

        print("对关节轨迹进行平滑处理...")

        trajectory_array = np.array(joint_trajectory)
        smoothed_trajectory = trajectory_array.copy()

        # 使用移动平均滤波
        window_size = max(3, int(len(joint_trajectory) * smoothing_factor))
        if window_size % 2 == 0:
            window_size += 1  # 确保窗口大小为奇数

        half_window = window_size // 2

        for i in range(half_window, len(trajectory_array) - half_window):
            for j in range(12):  # 12个关节
                window_values = trajectory_array[i - half_window:i + half_window + 1, j]
                smoothed_trajectory[i, j] = np.mean(window_values)

        if timestamps is None:
            timestamps = np.linspace(0, 5.0, len(joint_trajectory))

        # 确保timestamps是列表类型
        if isinstance(timestamps, np.ndarray):
            timestamps = timestamps.tolist()

        return smoothed_trajectory.tolist(), timestamps

    def interpolate_trajectory(self, joint_trajectory, timestamps, target_dt=0.01):
        """
        对轨迹进行时间插值，生成固定时间步长的轨迹

        Args:
            joint_trajectory (list): 关节角度轨迹
            timestamps (list): 时间戳列表
            target_dt (float): 目标时间步长（秒）

        Returns:
            tuple: (插值后的关节角度轨迹, 插值后的时间戳列表)
        """
        if len(joint_trajectory) < 2:
            return joint_trajectory, timestamps

        print(f"对轨迹进行时间插值，目标时间步长: {target_dt}s")

        # 确保timestamps是numpy数组
        if isinstance(timestamps, list):
            timestamps = np.array(timestamps)

        # 创建时间轴
        t_start = timestamps[0]
        t_end = timestamps[-1]
        t_interp = np.arange(t_start, t_end + target_dt, target_dt)

        # 对每个关节进行插值
        trajectory_array = np.array(joint_trajectory)
        interpolated_trajectory = []

        for t in t_interp:
            # 找到最近的两个时间点
            if t <= timestamps[0]:
                q = trajectory_array[0]
            elif t >= timestamps[-1]:
                q = trajectory_array[-1]
            else:
                # 线性插值
                for i in range(len(timestamps) - 1):
                    if timestamps[i] <= t <= timestamps[i + 1]:
                        t1, t2 = timestamps[i], timestamps[i + 1]
                        q1, q2 = trajectory_array[i], trajectory_array[i + 1]
                        alpha = (t - t1) / (t2 - t1)
                        q = q1 + alpha * (q2 - q1)
                        break
                else:
                    q = trajectory_array[-1]

            interpolated_trajectory.append(q)

        print(f"插值完成: {len(interpolated_trajectory)} 个轨迹点")
        return interpolated_trajectory, t_interp.tolist()

    def visualize_path_3d(self, path_poses, save_path=None):
        """
        3D可视化路径

        Args:
            path_poses (list): 路径点列表
            save_path (str): 保存路径，如果为None则不保存
        """
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # 提取路径点的位置
            positions = np.array([pose[:3, 3] for pose in path_poses])

            # 绘制路径
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                    'b-o', linewidth=2, markersize=4, label='规划路径')

            # 标记起点和终点
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
                       c='green', s=100, marker='s', label='起点')
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
                       c='red', s=100, marker='^', label='终点')

            # 绘制坐标轴
            for i in [0, len(positions) // 2, -1]:  # 起点、中点、终点
                pose = path_poses[i]
                pos = pose[:3, 3]
                rot = pose[:3, :3]

                # X轴 (红色)
                x_end = pos + 0.1 * rot[:, 0]
                ax.plot([pos[0], x_end[0]], [pos[1], x_end[1]], [pos[2], x_end[2]], 'r-', linewidth=2)

                # Y轴 (绿色)
                y_end = pos + 0.1 * rot[:, 1]
                ax.plot([pos[0], y_end[0]], [pos[1], y_end[1]], [pos[2], y_end[2]], 'g-', linewidth=2)

                # Z轴 (蓝色)
                z_end = pos + 0.1 * rot[:, 2]
                ax.plot([pos[0], z_end[0]], [pos[1], z_end[1]], [pos[2], z_end[2]], 'b-', linewidth=2)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('双臂协同操作路径规划')
            ax.legend()
            ax.grid(True)

            # 设置坐标轴比例
            ax.set_box_aspect([1, 1, 1])

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"路径可视化图已保存为: {save_path}")

            plt.show()

        except ImportError:
            print("无法导入matplotlib，跳过3D可视化")
        except Exception as e:
            print(f"3D可视化出错: {e}")


class DualArmSimulator:
    """
    双臂MuJoCo仿真器类

    该类封装了MuJoCo仿真功能，用于可视化路径执行
    """

    def __init__(self, xml_file):
        """
        初始化仿真器

        Args:
            xml_file (str): MuJoCo XML文件路径
        """
        self.xml_file = xml_file
        self.model = None
        self.data = None
        self._load_model()

    def _load_model(self):
        """加载MuJoCo模型"""
        try:
            import mujoco
            self.model = mujoco.MjModel.from_xml_path(self.xml_file)
            self.data = mujoco.MjData(self.model)
            print("MuJoCo模型加载成功")
        except ImportError:
            print("无法导入mujoco，仿真功能将不可用")
        except Exception as e:
            print(f"加载MuJoCo模型失败: {e}")

    def animate_trajectory(self, joint_trajectory, dt=0.05, realtime=True):
        """
        动画播放关节轨迹

        Args:
            joint_trajectory (list): 关节角度轨迹
            dt (float): 时间步长 (秒)
            realtime (bool): 是否实时播放
        """
        if self.model is None:
            print("MuJoCo模型未加载，无法进行仿真")
            return

        try:
            import mujoco
            import mujoco.viewer

            print(f"开始播放轨迹动画，共 {len(joint_trajectory)} 个关键帧")
            print("按空格暂停/继续，按ESC退出")

            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 设置相机位置和视角参数
                viewer.cam.distance = 4.0  # 相机距离：相机到观察目标的距离(米)，值越大视野越远
                viewer.cam.azimuth = -90  # 方位角：水平旋转角度(度)，0°=正前方，90°=右侧，180°=正后方，270°=左侧
                viewer.cam.elevation = -45  # 仰角：垂直角度(度)，正值=俯视，负值=仰视，0°=水平视角

                start_time = time.time()
                paused = False
                frame_idx = 0

                while viewer.is_running() and frame_idx < len(joint_trajectory):
                    current_time = time.time()

                    if not paused:
                        # 设置关节角度
                        self.data.qpos[:12] = joint_trajectory[frame_idx]

                        # 前向动力学
                        mujoco.mj_forward(self.model, self.data)

                        # 更新显示
                        viewer.sync()

                        frame_idx += 1

                        # 显示进度
                        if frame_idx % 10 == 0:
                            print(f"播放进度: {frame_idx}/{len(joint_trajectory)}")

                    # 控制播放速度
                    if realtime:
                        time.sleep(dt)

                # 保持最后一帧显示
                print("轨迹播放完成，按ESC退出")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.01)

        except ImportError:
            print("无法导入mujoco.viewer，使用备用显示方案")
            self._print_trajectory_summary(joint_trajectory)
        except Exception as e:
            print(f"仿真过程中发生错误: {e}")

    def _print_trajectory_summary(self, joint_trajectory):
        """打印轨迹摘要信息"""
        print("\n=== 轨迹摘要 ===")
        print(f"总帧数: {len(joint_trajectory)}")

        if joint_trajectory:
            # 分析关节角度变化范围
            trajectory_array = np.array(joint_trajectory)

            print("\n左臂关节角度范围 (度):")
            for i in range(6):
                min_angle = np.rad2deg(np.min(trajectory_array[:, i]))
                max_angle = np.rad2deg(np.max(trajectory_array[:, i]))
                print(f"  关节 {i + 1}: [{min_angle:.1f}, {max_angle:.1f}]")

            print("\n右臂关节角度范围 (度):")
            for i in range(6, 12):
                min_angle = np.rad2deg(np.min(trajectory_array[:, i]))
                max_angle = np.rad2deg(np.max(trajectory_array[:, i]))
                print(f"  关节 {i - 5}: [{min_angle:.1f}, {max_angle:.1f}]")


def demo_path_planning_system():
    """
    演示完整的路径规划系统
    """
    print("=" * 60)
    print("Jaka ZU12双臂机器人路径规划与仿真系统演示")
    print("=" * 60)

    # XML文件路径 - 使用Jaka ZU12配置
    xml_file = 'dual_jaka_zu12_scene.xml'

    try:
        # 1. 初始化路径规划器
        planner = DualArmPathPlanner(xml_file)

        # 2. 定义路径规划任务
        print("\n步骤1: 定义路径规划任务")

        # 起始位姿：home位姿
        start_pose = planner.T_W_M_home.copy()

        # 终点位姿：向右移动20cm，向上移动15cm
        end_pose = start_pose.copy()
        end_pose[:3, 3] += np.array([0.2, 0.1, 0.15])

        print(f"起始位置: {start_pose[:3, 3]}")
        print(f"目标位置: {end_pose[:3, 3]}")

        # 3. 规划不同类型的路径
        print("\n步骤2: 规划路径")

        # 选择路径类型
        path_type = "straight"  # 可选: "straight", "circular", "bezier"

        if path_type == "straight":
            path_poses = planner.plan_straight_line_path(start_pose, end_pose, num_points=15)
        elif path_type == "circular":
            # 以起始位置为中心规划圆形路径
            path_poses = planner.plan_circular_path(start_pose, radius=0.15, num_points=20)
        else:  # bezier
            path_poses = planner.plan_bezier_path(start_pose, end_pose, num_points=30)

        print(f"生成了 {len(path_poses)} 个路径点")

        # 4. 可视化路径
        print("\n步骤3: 可视化路径")
        planner.visualize_path_3d(path_poses, save_path='jaka_planned_path_3d.png')

        # 5. 求解逆运动学（带时间戳）
        print("\n步骤4: 求解逆运动学")
        total_execution_time = 8.0  # 总执行时间8秒
        joint_trajectory, timestamps, successful_solves = planner.solve_ik_for_path(
            path_poses, total_time=total_execution_time
        )

        if successful_solves == 0:
            print("错误：没有成功求解任何路径点的逆运动学")
            return

        print(f"轨迹时间范围: {timestamps[0]:.2f}s - {timestamps[-1]:.2f}s")

        # 6. 平滑轨迹
        print("\n步骤5: 平滑轨迹")
        smoothed_trajectory, smoothed_timestamps = planner.smooth_joint_trajectory(
            joint_trajectory, timestamps, smoothing_factor=0.2
        )

        # 7. 时间插值（可选）
        print("\n步骤6: 时间插值")
        interpolated_trajectory, interpolated_timestamps = planner.interpolate_trajectory(
            smoothed_trajectory, smoothed_timestamps, target_dt=0.02
        )

        print(f"插值后轨迹点数: {len(interpolated_trajectory)}")
        print(f"插值后时间步长: {interpolated_timestamps[1] - interpolated_timestamps[0]:.3f}s")

        # 8. MuJoCo仿真
        print("\n步骤7: MuJoCo仿真")
        simulator = DualArmSimulator(xml_file)
        simulator.animate_trajectory(interpolated_trajectory, dt=0.02, realtime=True)

        # 9. 保存结果（包含时间戳）
        print("\n步骤8: 保存结果")

        # 保存原始轨迹
        trajectory_data = {
            'joint_trajectory': np.array(joint_trajectory),
            'timestamps': np.array(timestamps),
            'successful_solves': successful_solves
        }
        np.save('jaka_joint_trajectory.npy', trajectory_data)

        # 保存平滑轨迹
        smoothed_data = {
            'joint_trajectory': np.array(smoothed_trajectory),
            'timestamps': np.array(smoothed_timestamps)
        }
        np.save('jaka_smoothed_trajectory.npy', smoothed_data)

        # 保存插值轨迹
        interpolated_data = {
            'joint_trajectory': np.array(interpolated_trajectory),
            'timestamps': np.array(interpolated_timestamps)
        }
        np.save('jaka_interpolated_trajectory.npy', interpolated_data)

        print("轨迹数据已保存:")
        print("  - jaka_joint_trajectory.npy (原始轨迹)")
        print("  - jaka_smoothed_trajectory.npy (平滑轨迹)")
        print("  - jaka_interpolated_trajectory.npy (插值轨迹)")

        # 10. 打印轨迹统计信息
        print("\n步骤9: 轨迹统计信息")
        print(f"原始轨迹点数: {len(joint_trajectory)}")
        print(f"平滑轨迹点数: {len(smoothed_trajectory)}")
        print(f"插值轨迹点数: {len(interpolated_trajectory)}")
        print(f"总执行时间: {interpolated_timestamps[-1] - interpolated_timestamps[0]:.2f}s")
        print(
            f"平均时间步长: {(interpolated_timestamps[-1] - interpolated_timestamps[0]) / len(interpolated_timestamps):.3f}s")

        print("\n" + "=" * 60)
        print("Jaka ZU12路径规划演示完成！")
        print("=" * 60)

    except Exception as e:
        print(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


def load_trajectory_data(filename):
    """
    加载保存的轨迹数据

    Args:
        filename (str): 轨迹数据文件名

    Returns:
        dict: 包含轨迹数据的字典
    """
    try:
        data = np.load(filename, allow_pickle=True).item()
        print(f"成功加载轨迹数据: {filename}")
        print(f"轨迹点数: {len(data['joint_trajectory'])}")
        if 'timestamps' in data:
            print(f"时间范围: {data['timestamps'][0]:.2f}s - {data['timestamps'][-1]:.2f}s")
        return data
    except Exception as e:
        print(f"加载轨迹数据失败: {e}")
        return None


def demo_load_and_play_trajectory():
    """
    演示加载和播放保存的轨迹
    """
    print("=" * 60)
    print("加载和播放保存的轨迹演示")
    print("=" * 60)

    # 加载插值后的轨迹数据
    trajectory_file = 'jaka_interpolated_trajectory.npy'
    data = load_trajectory_data(trajectory_file)

    if data is None:
        print("无法加载轨迹数据，请先运行路径规划演示")
        return

    # 初始化仿真器
    xml_file = 'dual_jaka_zu12_scene.xml'
    simulator = DualArmSimulator(xml_file)

    try:
        # 播放轨迹
        print("开始播放轨迹...")
        simulator.animate_trajectory(
            data['joint_trajectory'],
            dt=0.02,
            realtime=True
        )
    finally:
        simulator.close()


if __name__ == '__main__':
    # 可以选择运行不同的演示
    demo_path_planning_system()
    # demo_load_and_play_trajectory()  # 取消注释以运行轨迹播放演示
