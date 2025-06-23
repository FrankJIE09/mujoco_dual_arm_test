#!/usr/bin/env python3
"""
双臂机器人路径规划模块

该模块实现了基于贝塞尔曲线的路径规划功能，包括：
1. 贝塞尔曲线路径规划
2. 球面线性插值(SLERP)姿态插值
3. 逆运动学求解
4. 轨迹平滑处理
5. 路径点保存

作者: frank
日期: 2024年6月19日
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入字体配置
from font_config import init_chinese_font

# 导入双臂IK模块
from dual_arm_ik_modules.kinematic_chain import get_kinematics
from dual_arm_ik_modules.ik_solver import solve_dual_arm_ik

# 初始化中文字体支持
init_chinese_font(verbose=False)


class BezierPathPlanner:
    """
    基于贝塞尔曲线的双臂路径规划器
    
    该类封装了双臂协同操作的路径规划功能，使用：
    - 贝塞尔曲线进行位置插值
    - 球面线性插值(SLERP)进行姿态插值
    - 约束维护确保双臂闭环约束
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
        home_qpos_str = root.find(".//key[@name='home']").get('qpos')
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
        q1 = q[:6]  # 左臂6个关节角度
        q2 = q[6:]  # 右臂6个关节角度
        
        # ikpy正向运动学
        fk_q1 = np.concatenate(([0], q1, [0]))
        fk_q2 = np.concatenate(([0], q2, [0]))
        
        T_B1_E1 = self.left_chain.forward_kinematics(fk_q1)
        T_W_E1 = self.T_W_B1 @ T_B1_E1
        
        # 虚拟物体中心的变换关系（与IK求解器中的定义一致）
        T_E1_M = np.eye(4)
        T_E1_M[2, 3] = 0.24  # M在E1的Z轴正方向0.24m处
        
        T_W_M = T_W_E1 @ T_E1_M
        return T_W_M
    
    def plan_bezier_path(self, start_pose, end_pose, control_points=None, num_points=40):
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
        
        # 计算贝塞尔曲线位置
        t_values = np.linspace(0, 1, num_points)
        positions = self._bezier_curve(all_points, t_values)
        
        # 使用SLERP进行姿态插值
        orientations = self._slerp_interpolation(start_pose, end_pose, t_values)
        
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
                bernstein = self._binomial_coefficient(n, i) * (t**i) * ((1-t)**(n-i))
                point += bernstein * control_points[i]
            curve_points.append(point)
            
        return np.array(curve_points)
    
    def _slerp_interpolation(self, start_pose, end_pose, t_values):
        """
        使用球面线性插值(SLERP)进行姿态插值
        
        Args:
            start_pose (np.ndarray): 起始位姿 (4x4矩阵)
            end_pose (np.ndarray): 终点位姿 (4x4矩阵)
            t_values (np.ndarray): 参数值数组
            
        Returns:
            Rotation: 插值后的旋转序列
        """
        start_rot = Rotation.from_matrix(start_pose[:3, :3])
        end_rot = Rotation.from_matrix(end_pose[:3, :3])
        
        key_times = [0, 1]
        key_rotations = Rotation.concatenate([start_rot, end_rot])
        slerp = Slerp(key_times, key_rotations)
        
        return slerp(t_values)
    
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
    
    def solve_ik_for_path(self, path_poses, initial_q=None):
        """
        为路径上的每个点求解逆运动学
        
        Args:
            path_poses (list): 路径点列表
            initial_q (np.ndarray): 初始关节角度，如果为None则使用home位姿
            
        Returns:
            tuple: (关节角度轨迹列表, 成功求解的点数)
        """
        if initial_q is None:
            initial_q = self.home_q.copy()
            
        print(f"开始为 {len(path_poses)} 个路径点求解逆运动学...")
        
        joint_trajectory = []
        successful_solves = 0
        current_q = initial_q.copy()
        
        for i, target_pose in enumerate(path_poses):
            print(f"求解路径点 {i+1}/{len(path_poses)}")
            
            # 使用上一个成功的解作为初始猜测
            solution_q, _ = solve_dual_arm_ik(target_pose, current_q, self.kinematics_data)
            
            if solution_q is not None:
                joint_trajectory.append(solution_q.copy())
                current_q = solution_q.copy()  # 更新初始猜测
                successful_solves += 1
                
                # 验证解的质量
                actual_pose = self._compute_virtual_object_pose(solution_q)
                pos_error = np.linalg.norm(actual_pose[:3, 3] - target_pose[:3, 3])
                print(f"  成功! 位置误差: {pos_error:.4f}m")
            else:
                print(f"  失败! 在点 {i+1} 处未找到解")
                # 使用上一个成功的解作为备用
                if joint_trajectory:
                    joint_trajectory.append(joint_trajectory[-1].copy())
                else:
                    joint_trajectory.append(current_q.copy())
        
        print(f"逆运动学求解完成: {successful_solves}/{len(path_poses)} 个点成功求解")
        return joint_trajectory, successful_solves
    
    def smooth_joint_trajectory(self, joint_trajectory, smoothing_factor=0.2):
        """
        对关节角度轨迹进行平滑处理
        
        Args:
            joint_trajectory (list): 关节角度轨迹
            smoothing_factor (float): 平滑因子 (0-1)
            
        Returns:
            list: 平滑后的关节角度轨迹
        """
        if len(joint_trajectory) <= 2:
            return joint_trajectory
            
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
                window_values = trajectory_array[i-half_window:i+half_window+1, j]
                smoothed_trajectory[i, j] = np.mean(window_values)
        
        return smoothed_trajectory.tolist()
    
    def visualize_path_3d(self, path_poses, save_path=None):
        """
        3D可视化路径
        
        Args:
            path_poses (list): 路径点列表
            save_path (str): 保存路径，如果为None则不保存
        """
        try:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            # 提取路径点的位置
            positions = np.array([pose[:3, 3] for pose in path_poses])
            
            # 绘制贝塞尔曲线路径
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   'b-', linewidth=3, alpha=0.8, label='贝塞尔曲线路径')
            
            # 绘制路径点
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                      c='blue', s=20, alpha=0.6, label='路径点')
            
            # 标记起点和终点
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                      c='green', s=150, marker='s', label='起点', edgecolors='black', linewidths=2)
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                      c='red', s=150, marker='^', label='终点', edgecolors='black', linewidths=2)
            
            # 绘制关键点的坐标轴（起点、中点、终点）
            key_indices = [0, len(positions)//2, -1]
            for i in key_indices:
                pose = path_poses[i]
                pos = pose[:3, 3]
                rot = pose[:3, :3]
                
                # X轴 (红色)
                x_end = pos + 0.08 * rot[:, 0]
                ax.plot([pos[0], x_end[0]], [pos[1], x_end[1]], [pos[2], x_end[2]], 'r-', linewidth=2)
                
                # Y轴 (绿色)
                y_end = pos + 0.08 * rot[:, 1]
                ax.plot([pos[0], y_end[0]], [pos[1], y_end[1]], [pos[2], y_end[2]], 'g-', linewidth=2)
                
                # Z轴 (蓝色)
                z_end = pos + 0.08 * rot[:, 2]
                ax.plot([pos[0], z_end[0]], [pos[1], z_end[1]], [pos[2], z_end[2]], 'b-', linewidth=2)
            
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_zlabel('Z (m)', fontsize=12)
            ax.set_title('双臂协同操作 - 贝塞尔曲线路径规划', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # 设置坐标轴比例
            ax.set_box_aspect([1,1,1])
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"路径可视化图已保存为: {save_path}")
            
            plt.show()
            
        except ImportError:
            print("无法导入matplotlib，跳过3D可视化")
        except Exception as e:
            print(f"3D可视化出错: {e}")
    
    def save_trajectory_data(self, joint_trajectory, smoothed_trajectory, 
                           base_filename="trajectory"):
        """
        保存轨迹数据到文件
        
        Args:
            joint_trajectory (list): 原始关节轨迹
            smoothed_trajectory (list): 平滑后的关节轨迹
            base_filename (str): 基础文件名
        """
        # 保存原始轨迹
        raw_filename = f"{base_filename}_raw.npy"
        np.save(raw_filename, np.array(joint_trajectory))
        print(f"原始轨迹已保存为: {raw_filename}")
        
        # 保存平滑轨迹
        smooth_filename = f"{base_filename}_smooth.npy"
        np.save(smooth_filename, np.array(smoothed_trajectory))
        print(f"平滑轨迹已保存为: {smooth_filename}")
        
        # 保存轨迹摘要信息
        summary_filename = f"{base_filename}_summary.txt"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("双臂机器人轨迹摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"轨迹类型: 贝塞尔曲线路径\n")
            f.write(f"原始轨迹点数: {len(joint_trajectory)}\n")
            f.write(f"平滑轨迹点数: {len(smoothed_trajectory)}\n")
            f.write(f"关节数量: 12 (左臂6个 + 右臂6个)\n")
            f.write("\n")
            
            # 分析关节角度变化范围
            if joint_trajectory:
                trajectory_array = np.array(joint_trajectory)
                f.write("左臂关节角度范围 (度):\n")
                for i in range(6):
                    min_angle = np.rad2deg(np.min(trajectory_array[:, i]))
                    max_angle = np.rad2deg(np.max(trajectory_array[:, i]))
                    f.write(f"  关节 {i+1}: [{min_angle:.1f}, {max_angle:.1f}]\n")
                
                f.write("\n右臂关节角度范围 (度):\n")
                for i in range(6, 12):
                    min_angle = np.rad2deg(np.min(trajectory_array[:, i]))
                    max_angle = np.rad2deg(np.max(trajectory_array[:, i]))
                    f.write(f"  关节 {i-5}: [{min_angle:.1f}, {max_angle:.1f}]\n")
        
        print(f"轨迹摘要已保存为: {summary_filename}")


def demo_bezier_path_planning():
    """
    演示贝塞尔曲线路径规划
    """
    print("=" * 60)
    print("双臂机器人贝塞尔曲线路径规划演示")
    print("=" * 60)
    
    # XML文件路径
    xml_file = 'dual_elfin15_scene_clean.xml'
    
    try:
        # 1. 初始化路径规划器
        planner = BezierPathPlanner(xml_file)
        
        # 2. 定义路径规划任务
        print("\n步骤1: 定义路径规划任务")
        
        # 起始位姿：home位姿
        start_pose = planner.T_W_M_home.copy()
        
        # 终点位姿：向右移动30cm，向前移动10cm，向上移动20cm
        end_pose = start_pose.copy()
        end_pose[:3, 3] += np.array([0.3, 0.1, 0.2])
        
        print(f"起始位置: {start_pose[:3, 3]}")
        print(f"目标位置: {end_pose[:3, 3]}")
        
        # 3. 规划贝塞尔曲线路径
        print("\n步骤2: 规划贝塞尔曲线路径")
        path_poses = planner.plan_bezier_path(start_pose, end_pose, num_points=40)
        print(f"生成了 {len(path_poses)} 个路径点")
        
        # 4. 可视化路径
        print("\n步骤3: 可视化路径")
        planner.visualize_path_3d(path_poses, save_path='bezier_path_3d.png')
        
        # 5. 求解逆运动学
        print("\n步骤4: 求解逆运动学")
        joint_trajectory, successful_solves = planner.solve_ik_for_path(path_poses)
        
        if successful_solves == 0:
            print("错误：没有成功求解任何路径点的逆运动学")
            return None, None, None
        
        # 6. 平滑轨迹
        print("\n步骤5: 平滑轨迹")
        smoothed_trajectory = planner.smooth_joint_trajectory(joint_trajectory, smoothing_factor=0)
        
        # 7. 保存轨迹数据
        print("\n步骤6: 保存轨迹数据")
        planner.save_trajectory_data(joint_trajectory, smoothed_trajectory, "bezier_trajectory")
        
        print("\n" + "=" * 60)
        print("贝塞尔曲线路径规划完成！")
        print("=" * 60)
        
        return joint_trajectory, smoothed_trajectory, planner
        
    except Exception as e:
        print(f"路径规划过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == '__main__':
    demo_bezier_path_planning() 