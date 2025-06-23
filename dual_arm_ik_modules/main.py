"""
双臂机器人逆运动学求解主程序

这个模块整合了所有功能模块，实现完整的双臂逆运动学求解流程。
主要功能包括：
1. 解析XML文件并创建运动学链
2. 定义目标位姿
3. 求解逆运动学
4. 验证结果
5. 生成可视化

作者: [您的姓名]
日期: [创建日期]
"""

import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

# 导入自定义模块
from .kinematic_chain import get_kinematics
from .ik_solver import solve_dual_arm_ik
from .visualization import (
    plot_iteration_history,
    show_mujoco_iteration_process,
    show_mujoco_home_pose,
    print_optimization_statistics
)


def main():
    """
    主程序：演示双臂运动学求解的完整流程

    包括以下步骤：
    1. 解析XML文件并创建运动学链
    2. 定义目标位姿
    3. 求解逆运动学
    4. 验证结果
    5. 生成可视化
    """
    xml_file = '../dual_elfin15_scene_clean.xml'

    # --- 1. 设置阶段 ---
    print("正在解析XML文件并创建运动学链...")
    kinematics_data = get_kinematics(xml_file)
    left_chain, right_chain, T_W_B1, T_W_B2 = kinematics_data
    print("运动学链创建完成。")
    print(f"左臂活动连杆掩码: {left_chain.active_links_mask}")
    print(f"右臂活动连杆掩码: {right_chain.active_links_mask}")

    # 运动学的起点定义为两个基座连线的中点。
    # 在这个XML文件中，两个基座位于同一位置，所以中点就是那个位置。
    # 数学公式: p_origin = (p_B1 + p_B2) / 2
    p_B1 = T_W_B1[:3, 3]  # 左臂基座位置
    p_B2 = T_W_B2[:3, 3]  # 右臂基座位置
    kinematic_origin = (p_B1 + p_B2) / 2  # 计算中点
    print(f"运动学原点 (基座中点): {kinematic_origin}")

    # --- 2. 定义目标位姿 ---
    # 从XML的'home'关键帧读取关节角度，计算虚拟物体中点M的实际位姿作为目标
    tree = ET.parse(xml_file)
    root = tree.getroot()
    home_qpos_str = root.find(".//key[@name='home']").get('qpos')
    home_q = np.fromstring(home_qpos_str, sep=' ')

    print(f"\n使用'home'关键帧的关节角度: {home_q}")

    # 使用home关键帧的关节角度进行正向运动学，计算虚拟物体中点M的实际位姿
    home_q1 = home_q[:6]  # 左臂6个关节角度
    home_q2 = home_q[6:]  # 右臂6个关节角度

    # ikpy正向运动学需要包含固定连杆的角度向量
    fk_home_q1 = np.concatenate(([0], home_q1, [0]))
    fk_home_q2 = np.concatenate(([0], home_q2, [0]))

    # 计算home位姿下的双臂末端位姿
    T_B1_E1_home = left_chain.forward_kinematics(fk_home_q1)
    T_B2_E2_home = right_chain.forward_kinematics(fk_home_q2)

    T_W_E1_home = T_W_B1 @ T_B1_E1_home
    T_W_E2_home = T_W_B2 @ T_B2_E2_home

    # 计算虚拟物体中点M在home位姿下的实际位姿
    # 使用左臂末端计算M的位姿
    T_E1_M = np.eye(4)
    T_E1_M[2, 3] = 0.24  # M在E1的Z轴正方向0.24m处
    T_W_M_home = T_W_E1_home @ T_E1_M

    # 使用这个实际位姿作为目标位姿
    target_pose_M = T_W_M_home.copy()

    print(f"\nHome位姿下虚拟物体中点的实际位姿 (作为目标位姿):")
    print(f"位置: {target_pose_M[:3, 3]}")
    print(f"旋转矩阵:")
    print(target_pose_M[:3, :3])

    # 验证闭环约束在home位姿下是否满足
    T_E1_E2_home = np.linalg.inv(T_W_E1_home) @ T_W_E2_home
    print(f"\nHome位姿下两臂末端的相对位姿:")
    print(f"相对位置: {T_E1_E2_home[:3, 3]}")
    print(f"相对旋转矩阵:")
    print(T_E1_E2_home[:3, :3])

    # --- 2.5. 展示Home位姿 ---
    print("\n正在展示Home关键帧下的双臂位姿...")
    show_mujoco_home_pose(xml_file, home_q, target_pose_M, T_W_E1_home, T_W_E2_home)

    # --- 3. 求解阶段 ---
    # 使用home关键帧的关节角度作为优化的初始猜测值
    initial_q = home_q.copy()

    print(f"\n使用'home'关键帧的关节角度作为初始猜测值: {initial_q}")

    print("\n正在求解双臂逆运动学...")
    solution_q, iteration_history = solve_dual_arm_ik(target_pose_M, initial_q, kinematics_data)

    # --- 4. 验证阶段 ---
    if solution_q is not None:
        print("\n验证求解结果:")
        q1_sol = solution_q[:6]  # 左臂解
        q2_sol = solution_q[6:]  # 右臂解
        print(f"左臂关节角度解 (度): {np.rad2deg(q1_sol)}")
        print(f"右臂关节角度解 (度): {np.rad2deg(q2_sol)}")

        # 使用找到的解进行正向运动学，计算实际到达的位姿
        # 数学公式: T_B1_E1 = ∏(i=1 to 6) T_i(q_i_sol)
        fk_q1_sol = np.concatenate(([0], q1_sol, [0]))
        fk_q2_sol = np.concatenate(([0], q2_sol, [0]))

        T_B1_E1_sol = left_chain.forward_kinematics(fk_q1_sol)
        T_B2_E2_sol = right_chain.forward_kinematics(fk_q2_sol)

        T_W_E1_sol = T_W_B1 @ T_B1_E1_sol
        T_W_E2_sol = T_W_B2 @ T_B2_E2_sol

        # 注意：这里的 T_E1_M 必须与上面 solve_dual_arm_ik 中定义的完全一致
        # 数学公式: T_W_M_sol = T_W_E1_sol @ T_E1_M
        T_E1_M = np.eye(4)
        T_E1_M[2, 3] = 0.24
        T_W_M_sol = T_W_E1_sol @ T_E1_M

        print("\n求解得到的虚拟物体中点位姿:")
        print(T_W_M_sol)

        # 计算实际位姿与目标位姿的差距
        # 数学公式: pos_diff = ||p_M_sol - p_target||
        pos_diff = np.linalg.norm(T_W_M_sol[:3, 3] - target_pose_M[:3, 3])
        print(f"\n与目标位置的距离: {pos_diff:.6f} 米")

        # 验证闭环约束是否仍然满足
        # 注意：这里的 T_E1_M 和 T_E2_M 的定义也必须与上面 solve_dual_arm_ik 中定义的完全一致
        # 数学公式: T_E1_E2_sol = inv(T_W_E1_sol) @ T_W_E2_sol
        T_E1_M = np.eye(4)
        T_E1_M[2, 3] = 0.24
        T_E2_M = np.eye(4)
        T_E2_M[:3, :3] = Rotation.from_euler('y', 180, degrees=True).as_matrix()
        T_E2_M[:3, 3] = np.array([0, 0, 0.24])
        T_E1_E2_target = T_E1_M @ np.linalg.inv(T_E2_M)
        T_E1_E2_sol = np.linalg.inv(T_W_E1_sol) @ T_W_E2_sol

        # 数学公式: constraint_pos_diff = ||p_E1_E2_sol - p_E1_E2_target||
        constraint_pos_diff = np.linalg.norm(T_E1_E2_sol[:3, 3] - T_E1_E2_target[:3, 3])
        print(f"闭环约束位置误差: {constraint_pos_diff:.6f} 米")

        # 显示优化结果统计
        print_optimization_statistics(
            iteration_history, solution_q, initial_q,
            target_pose_M, T_W_M_sol, pos_diff, constraint_pos_diff
        )

        # --- 4.5. 可视化迭代历史 ---
        print("\n正在可视化迭代历史...")
        plot_iteration_history(iteration_history)

        # --- 5. 可视化阶段 ---
        print("\n正在使用MuJoCo viewer展示结果...")
        show_mujoco_iteration_process(xml_file, iteration_history, solution_q)

    else:
        print("逆运动学求解失败！")


if __name__ == '__main__':
    main()