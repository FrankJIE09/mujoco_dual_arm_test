"""
单臂逆运动学演示程序

该模块实现了：
1. 输入xyz位置和rpy姿态
2. 使用scipy处理旋转变换
3. 左右臂分别计算逆运动学
4. 在MuJoCo中展示结果

作者: frank
日期: 2024年12月
"""

import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation
import ikpy.chain
import ikpy.link
import time
import random

# 导入现有的运动学模块
from dual_arm_kinematics import get_kinematics


class SingleArmIKSolver:
    """
    单臂逆运动学求解器
    
    支持左右臂独立的逆运动学计算和MuJoCo可视化
    """
    
    def __init__(self, xml_file='dual_jaka_zu12_scene.xml'):
        """
        初始化逆运动学求解器
        
        Args:
            xml_file (str): MuJoCo XML场景文件路径
        """
        self.xml_file = xml_file
        
        print("正在初始化单臂逆运动学求解器...")
        
        # 获取运动学模型
        self.left_chain, self.right_chain, self.T_W_B1, self.T_W_B2 = get_kinematics(xml_file)
        
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        
        print("求解器初始化完成")
        print(f"左臂基座位置: {self.T_W_B1[:3, 3]}")
        print(f"右臂基座位置: {self.T_W_B2[:3, 3]}")
    
    def solve_ik(self, target_xyz, target_rpy_deg, arm='left', initial_guess=None):
        """
        求解单臂逆运动学
        
        Args:
            target_xyz (list): 目标位置 [x, y, z] (基座坐标系)
            target_rpy_deg (list): 目标姿态 [roll, pitch, yaw] (度)
            arm (str): 'left' 或 'right'
            initial_guess (np.ndarray): 初始关节角度猜测值
            
        Returns:
            dict: 求解结果，包含成功标志、关节角度、误差等信息
        """
        print(f"\n=== {arm.upper()}臂逆运动学求解 ===")
        print(f"目标位置 (基座坐标): {target_xyz}")
        print(f"目标姿态 (度): {target_rpy_deg}")
        
        # 选择对应的运动学链
        if arm == 'left':
            chain = self.left_chain
        else:
            chain = self.right_chain
        
        # 转换RPY角度到旋转矩阵
        rpy_rad = np.deg2rad(target_rpy_deg)
        rotation = Rotation.from_euler('XYZ', rpy_rad)
        target_orientation = rotation.as_matrix()
        
        # 目标位置（基座坐标系）
        target_position = np.array(target_xyz)
        
        # 设置初始猜测值
        if initial_guess is None:
            # 使用当前关节角度作为初始猜测
            if arm == 'left':
                current_q = self.data.qpos[:6]
            else:
                current_q = self.data.qpos[6:12]
            initial_guess = np.concatenate(([0], current_q, [0]))
        else:
            initial_guess = np.concatenate(([0], initial_guess, [0]))
        
        print(f"初始猜测关节角度: {np.rad2deg(initial_guess[1:-1])}")
        
        try:
            # 使用更好的初始猜测来改善姿态精度
            # 根据目标姿态调整初始猜测
            if target_rpy_deg[1] > 45:  # 如果pitch角度大，可能是向下指向
                better_initial_guess = np.array([0, -45, 90, 0, -45, 0]) * np.pi/180
            elif abs(target_rpy_deg[0]) > 90:  # 如果roll角度大，可能需要特殊配置
                better_initial_guess = np.array([90, -30, 60, 30, 0, 0]) * np.pi/180
            else:  # 默认工作姿态
                better_initial_guess = np.array([0, -30, 60, 30, 0, 0]) * np.pi/180
            
            # 构造完整的关节角度向量（添加固定关节）
            full_initial_guess = np.concatenate(([0], better_initial_guess, [0]))
            
            print(f"使用改进的初始猜测: {np.rad2deg(better_initial_guess)}")
            
            # 使用ikpy求解逆运动学
            solution_full = chain.inverse_kinematics(
                target_position=target_position,
                target_orientation=target_orientation,
                initial_position=full_initial_guess,
                orientation_mode = "all"
            )
            
            # 提取关节角度（去除首尾的虚拟关节）
            solution_joints = solution_full[1:-1]
            
            # 验证解的有效性
            fk_result = chain.forward_kinematics(solution_full)
            actual_position = fk_result[:3, 3]
            actual_orientation = fk_result[:3, :3]
            
            # 计算位置误差
            position_error = np.linalg.norm(actual_position - target_position)
            
            # 改进的姿态误差计算
            # 方法1：使用旋转矩阵的Frobenius范数
            orientation_error_frobenius = np.linalg.norm(target_orientation - actual_orientation, 'fro')
            
            # 方法2：使用旋转角度差（更准确的方法）
            R_error = target_orientation.T @ actual_orientation
            trace_R = np.trace(R_error)
            # 确保trace在有效范围内
            trace_R = np.clip(trace_R, -1, 3)
            orientation_error_angle = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))
            
            # 方法3：使用四元数比较
            target_quat = Rotation.from_matrix(target_orientation).as_quat()
            actual_quat = Rotation.from_matrix(actual_orientation).as_quat()
            # 处理四元数的符号问题（q和-q代表相同旋转）
            if np.dot(target_quat, actual_quat) < 0:
                actual_quat = -actual_quat
            quat_error = np.linalg.norm(target_quat - actual_quat)
            
            print(f"求解成功!")
            print(f"关节角度解 (弧度): {solution_joints}")
            print(f"关节角度解 (度): {np.rad2deg(solution_joints)}")
            print(f"位置误差: {position_error:.6f} 米")
            
            # 详细的姿态误差分析
            print(f"\n=== 姿态误差分析 ===")
            print(f"方法1 - 矩阵Frobenius范数误差: {orientation_error_frobenius:.6f}")
            print(f"方法2 - 旋转角度误差: {np.rad2deg(orientation_error_angle):.6f} 度")
            print(f"方法3 - 四元数误差: {quat_error:.6f}")
            
            # 转换实际姿态回RPY角度用于验证
            actual_rotation = Rotation.from_matrix(actual_orientation)
            actual_rpy_deg = actual_rotation.as_euler('XYZ', degrees=True)
            
            print(f"\n=== 姿态对比 ===")
            print(f"目标位置: {target_position}")
            print(f"实际位置: {actual_position}")
            print(f"目标姿态 (度): {target_rpy_deg}")
            print(f"实际姿态 (度): {actual_rpy_deg}")
            
            # 检查是否是等效旋转
            print(f"\n=== 旋转等效性检查 ===")
            print(f"目标四元数: {target_quat}")
            print(f"实际四元数: {actual_quat}")
            
            # 验证旋转是否等效（通过检查同一向量的变换结果）
            test_vector = np.array([1, 0, 0])  # 测试向量
            target_transformed = target_orientation @ test_vector
            actual_transformed = actual_orientation @ test_vector
            vector_diff = np.linalg.norm(target_transformed - actual_transformed)
            print(f"测试向量变换差异: {vector_diff:.6f}")
            
            return {
                'success': True,
                'joint_angles': solution_joints,
                'joint_angles_deg': np.rad2deg(solution_joints),
                'position_error': position_error,
                'orientation_error_angle': orientation_error_angle,
                'orientation_error_frobenius': orientation_error_frobenius,
                'quat_error': quat_error,
                'vector_transform_error': vector_diff,
                'actual_position': actual_position,
                'actual_rpy_deg': actual_rpy_deg,
                'target_position': target_position,
                'target_rpy_deg': target_rpy_deg,
                'target_quat': target_quat,
                'actual_quat': actual_quat
            }
            
        except Exception as e:
            print(f"逆运动学求解失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'target_position': target_position,
                'target_rpy_deg': target_rpy_deg
            }
    
    def get_forward_kinematics_base(self, joint_angles, arm='left'):
        """
        计算正向运动学（基座坐标系）
        
        Args:
            joint_angles (np.ndarray): 关节角度 (6个关节)
            arm (str): 'left' 或 'right'
            
        Returns:
            tuple: (position, rpy_deg) 位置和RPY姿态
        """
        # 选择对应的运动学链
        if arm == 'left':
            chain = self.left_chain
        else:
            chain = self.right_chain
        
        # 构造完整的关节角度向量（包含虚拟关节）
        full_angles = np.concatenate(([0], joint_angles, [0]))
        
        # 计算正向运动学
        T_base_end = chain.forward_kinematics(full_angles)
        
        # 提取位置和姿态
        position = T_base_end[:3, 3]
        rotation_matrix = T_base_end[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        rpy_deg = rotation.as_euler('XYZ', degrees=True)
        
        return position, rpy_deg
    
    def visualize_result(self, results_dict, display_time=10.0):
        """
        在MuJoCo中可视化逆运动学结果
        
        Args:
            results_dict (dict): 包含左臂和/或右臂求解结果的字典
            display_time (float): 显示时间（秒）
        """
        print(f"\n=== MuJoCo可视化 ===")
        print(f"显示时间: {display_time}秒")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            
            while viewer.is_running() and (time.time() - start_time) < display_time:
                # 持续设置关节角度以维持姿态
                if 'left' in results_dict and results_dict['left']['success']:
                    q_left = results_dict['left']['joint_angles']
                    for i in range(6):
                        self.data.qvel[i] = 10*(q_left[i]-self.data.qpos[i])

                if 'right' in results_dict and results_dict['right']['success']:
                    q_right = results_dict['right']['joint_angles']
                    for i in range(6):
                        self.data.qvel[i+6] = 10*(q_right[i]-self.data.qpos[i+6])

                        # self.data.qpos[i + 6] = q_right[i]
                
                # 执行仿真步
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                time.sleep(0.01)
        
        print("可视化结束")


def main():
    """
    主函数：演示单臂逆运动学功能
    """
    print("JAKA ZU12 单臂逆运动学演示程序")
    print("="*40)
    
    # 创建求解器
    solver = SingleArmIKSolver()
    
    # 演示用例：求解左臂逆运动学
    print("\n=== 单臂逆运动学演示 ===")
    
    # 定义目标位姿（基座坐标系）
    target_xyz = [-0.1, -0.4, 0.5]   # 位置
    target_rpy = [180, 0, 0]        # 姿态（度）
    
    print(f"目标位置: {target_xyz}")
    print(f"目标姿态: {target_rpy}")
    
    # 求解左臂逆运动学
    left_result = solver.solve_ik(target_xyz, target_rpy, arm='left')
    
    # 求解右臂逆运动学（相对应的位置）
    right_target_xyz = [0.1, -0.4, 0.5]  # 右臂对应位置
    right_target_rpy = [180, 0, 0]        # 姿态（度）

    right_result = solver.solve_ik(right_target_xyz, right_target_rpy, arm='right')
    
    # 收集结果
    results = {}
    if left_result['success']:
        results['left'] = left_result
        print(f"\n左臂求解成功，位置误差: {left_result['position_error']:.6f} 米")
    
    if right_result['success']:
        results['right'] = right_result
        print(f"右臂求解成功，位置误差: {right_result['position_error']:.6f} 米")
    
    # 可视化结果
    if results:
        print("\n启动MuJoCo可视化...")
        solver.visualize_result(results, display_time=800.0)
    else:
        print("没有成功的求解结果可显示")
    
    print("\n程序结束")


if __name__ == "__main__":
    main()