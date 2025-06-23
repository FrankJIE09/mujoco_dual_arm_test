# -*- coding: utf-8 -*-
"""
@File    :   demo_velocity_control.py
@Time    :   2024/07/28
@Author  :   Your Name
@Version :   1.0
@Desc    :   演示和对比关节控制模式: 直接位置设置 vs. 速度伺服控制。
"""

import os
import numpy as np
from mujoco_simulator import DualArmSimulator, load_trajectory_from_file
from path_planner import BezierPathPlanner
from font_config import init_chinese_font

def get_user_choice():
    """提供一个菜单供用户选择控制模式和参数"""
    print("\n" + "="*50)
    print("请选择一个仿真选项:")
    print("  1. 运动学模式: 直接设置关节位置 (Position Control)")
    print("  2. 动力学模式: 速度伺服跟踪 (Velocity Servo Control)")
    print("  3. 重新生成轨迹 (Regenerate Trajectory)")
    print("  4. 退出 (Exit)")
    print("="*50)
    
    choice = input("请输入您的选择 [1-4]: ")
    
    kp = 10.0  # 默认Kp值
    if choice == '2':
        try:
            kp_input = input("请输入P控制器增益 Kp (直接回车使用默认值 10.0): ")
            if kp_input.strip():
                kp = float(kp_input)
        except ValueError:
            print("无效输入，将使用默认值 Kp=10.0")
            
    return choice, kp

def main():
    """主函数，运行整个演示流程"""
    # 初始化中文字体，确保matplotlib绘图时能正确显示中文
    init_chinese_font()
    
    # --- 配置文件路径 ---
    MODEL_XML_PATH = 'dual_elfin15_scene_clean_velocity.xml'
    TRAJECTORY_FILE = 'dual_arm_trajectory_smoothed.npy'

    # --- 检查轨迹文件是否存在 ---
    if not os.path.exists(TRAJECTORY_FILE):
        print(f"⚠️ 轨迹文件 {TRAJECTORY_FILE} 不存在。将首先为您生成一个新的轨迹。")
        run_path_planner()

    # --- 加载轨迹 ---
    print(f"\n--- 加载轨迹文件: {TRAJECTORY_FILE} ---")
    joint_trajectory = load_trajectory_from_file(TRAJECTORY_FILE)
    if joint_trajectory is None:
        return # 如果加载失败则退出

    # --- 初始化仿真器 ---
    print("\n--- 初始化MuJoCo仿真器 ---")
    simulator = DualArmSimulator(MODEL_XML_PATH)
    if simulator.model is None:
        print("❌ 仿真器初始化失败，程序退出。")
        return
        
    # --- 设置相机视角 ---
    simulator.set_camera_params(distance=3.5, azimuth=90, elevation=-20)

    # --- 用户交互循环 ---
    while True:
        choice, kp_value = get_user_choice()
        
        if choice == '1':
            print("\n--- 启动运动学仿真: 直接位置控制 ---")
            simulator.animate_trajectory(
                joint_trajectory, 
                control_mode='position',
                dt=0.02  # 运动学模式下dt可以稍大
            )
        elif choice == '2':
            print(f"\n--- 启动动力学仿真: 速度伺服控制 (Kp={kp_value}) ---")
            simulator.animate_trajectory(
                joint_trajectory, 
                control_mode='velocity_servo', 
                kp=kp_value,
                dt=0.02  # 动力学模式下，dt决定了目标点更新的频率
            )
        elif choice == '3':
            run_path_planner()
            print(f"\n--- 重新加载轨迹文件: {TRAJECTORY_FILE} ---")
            joint_trajectory = load_trajectory_from_file(TRAJECTORY_FILE)
            if joint_trajectory is None:
                print("加载新轨迹失败，请检查路径规划器是否成功运行。")
                continue

        elif choice == '4':
            print("👋 程序退出。")
            break
        else:
            print("无效选择，请输入 1, 2, 3 或 4。")

def run_path_planner():
    """运行路径规划器以生成轨迹"""
    print("\n" + "#"*50)
    print("### 开始运行路径规划器 ###")
    print("#"*50)
    
    # 初始化规划器
    planner = BezierPathPlanner(xml_file='dual_elfin15_scene_clean_velocity.xml')
    
    # 1. 定义起始和结束位姿
    start_pose = planner.T_W_M_home
    end_pose = start_pose.copy()
    end_pose[0, 3] += 0.5  # X轴方向移动0.5米
    end_pose[1, 3] -= 0.3  # Z轴方向下降0.3米
    end_pose[2, 3] -= 0.3  # Z轴方向下降0.3米

    # 2. 规划贝塞尔路径
    path_poses = planner.plan_bezier_path(start_pose, end_pose, num_points=100)
    
    # 3. 为路径求解IK
    raw_trajectory, success_count = planner.solve_ik_for_path(path_poses)
    
    if success_count < len(path_poses) * 0.8:
        print("⚠️ 逆运动学求解成功率过低，可能导致轨迹不连贯。")

    # 4. 平滑关节轨迹
    smoothed_trajectory = planner.smooth_joint_trajectory(np.array(raw_trajectory))
    
    # 5. 可视化结果
    planner.visualize_path_3d(path_poses, save_path="path_visualization.png")

    # 6. 保存轨迹数据
    output_file = "dual_arm_trajectory"
    planner.save_trajectory_data(
        np.array(raw_trajectory),
        np.array(smoothed_trajectory),
        base_filename=output_file
    )
    
    print("\n" + "#"*50)
    print("### 路径规划完成 ###")
    print(f"平滑后的轨迹已保存到: {output_file}_smoothed.npy")
    print(f"可视化图像已保存到: path_visualization.png")
    print("#"*50)

if __name__ == '__main__':
    main() 