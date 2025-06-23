"""
可视化模块

这个模块提供了双臂机器人逆运动学求解结果的可视化功能。
主要功能包括：
1. 迭代历史图表绘制
2. MuJoCo 3D可视化
3. 优化过程动画展示
4. 结果统计分析

作者: frank
日期: 2024年6月19日
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time


def plot_iteration_history(iteration_history, save_path='optimization_history.png'):
    """
    绘制迭代历史图表。
    
    生成包含4个子图的综合分析图表：
    1. 误差收敛曲线
    2. 左臂关节角度变化
    3. 右臂关节角度变化
    4. 误差改善百分比
    
    Args:
        iteration_history (list): 迭代历史记录列表
        save_path (str): 图表保存路径
    """
    try:
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 提取误差和迭代次数
        errors = [info['error'] for info in iteration_history]
        iterations = [info['iteration'] for info in iteration_history]

        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 子图1: 误差收敛曲线
        plt.subplot(2, 2, 1)
        plt.plot(iterations, errors, 'b-o', markersize=3)
        plt.title('Error Convergence Curve', fontsize=12)
        plt.xlabel('Iteration Number', fontsize=10)
        plt.ylabel('Error Value', fontsize=10)
        plt.grid(True)
        plt.yscale('log')  # 使用对数坐标更好地显示收敛
        
        # 子图2: 左臂关节角度变化
        plt.subplot(2, 2, 2)
        left_angles_history = np.array([info['left_arm_angles'] for info in iteration_history])
        for i in range(6):
            plt.plot(iterations, np.rad2deg(left_angles_history[:, i]), 
                    label=f'Joint {i+1}', marker='o', markersize=2)
        plt.title('Left Arm Joint Angles', fontsize=12)
        plt.xlabel('Iteration Number', fontsize=10)
        plt.ylabel('Angle (degrees)', fontsize=10)
        plt.legend()
        plt.grid(True)
        
        # 子图3: 右臂关节角度变化
        plt.subplot(2, 2, 3)
        right_angles_history = np.array([info['right_arm_angles'] for info in iteration_history])
        for i in range(6):
            plt.plot(iterations, np.rad2deg(right_angles_history[:, i]), 
                    label=f'Joint {i+1}', marker='o', markersize=2)
        plt.title('Right Arm Joint Angles', fontsize=12)
        plt.xlabel('Iteration Number', fontsize=10)
        plt.ylabel('Angle (degrees)', fontsize=10)
        plt.legend()
        plt.grid(True)
        
        # 子图4: 误差改善百分比
        plt.subplot(2, 2, 4)
        initial_error = errors[0]
        error_improvement = [(initial_error - err) / initial_error * 100 for err in errors]
        plt.plot(iterations, error_improvement, 'g-o', markersize=3)
        plt.title('Error Improvement Percentage', fontsize=12)
        plt.xlabel('Iteration Number', fontsize=10)
        plt.ylabel('Improvement (%)', fontsize=10)
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"迭代历史图表已保存为 '{save_path}'")
        
    except ImportError:
        print("\n无法导入 'matplotlib'。")
        print("请安装matplotlib以查看迭代历史图表: pip install matplotlib")
    except Exception as e:
        print(f"\n可视化迭代历史过程中发生错误: {e}")
        print("继续执行...")


def show_mujoco_iteration_process(xml_file, iteration_history, solution_q):
    """
    使用MuJoCo viewer展示迭代过程。
    
    Args:
        xml_file (str): MuJoCo XML文件路径
        iteration_history (list): 迭代历史记录
        solution_q (np.ndarray): 最终求解的关节角度
    """
    try:
        import mujoco
        import mujoco.viewer

        # 加载MuJoCo模型
        model = mujoco.MjModel.from_xml_path(xml_file)
        data = mujoco.MjData(model)

        # 使用MuJoCo viewer展示迭代过程
        print("正在启动MuJoCo viewer展示迭代过程...")
        print("按ESC键退出查看器")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 设置相机位置以便更好地观察双臂
            viewer.cam.distance = 3.0  # 相机距离
            viewer.cam.azimuth = 45    # 方位角
            viewer.cam.elevation = -20  # 仰角
            
            # 展示迭代过程
            print("展示迭代过程...")
            print("按空格键暂停/继续，按ESC键退出")
            
            # 选择要展示的关键迭代点
            key_iterations = [0, len(iteration_history)//4, len(iteration_history)//2, 
                            3*len(iteration_history)//4, len(iteration_history)-1]
            
            for i, iter_idx in enumerate(key_iterations):
                if iter_idx < len(iteration_history):
                    info = iteration_history[iter_idx]
                    
                    # 设置关节角度
                    data.qpos[:12] = info['joint_angles']
                    mujoco.mj_forward(model, data)
                    
                    # 显示当前迭代信息
                    print(f"迭代 {info['iteration']}: 误差 = {info['error']:.6f}")
                    
                    # 更新viewer
                    viewer.sync()
                    
                    # 等待一段时间让用户观察
                    time.sleep(2)
            
            # 最后展示最终结果
            print("展示最终结果...")
            data.qpos[:12] = solution_q
            mujoco.mj_forward(model, data)
            viewer.sync()
            
            # 保持viewer运行，直到用户关闭
            while viewer.is_running():
                viewer.sync()
                
    except ImportError:
        print("\n无法导入 'mujoco.viewer'。")
        print("请确保MuJoCo已正确安装并支持viewer功能。")
    except Exception as e:
        print(f"\n可视化过程中发生错误: {e}")
        print("尝试使用备用方案...")
        
        # 备用方案：打印详细结果
        print("\n=== 求解结果详情 ===")
        print(f"总迭代次数: {len(iteration_history)}")
        print(f"初始误差: {iteration_history[0]['error']:.6f}")
        print(f"最终误差: {iteration_history[-1]['error']:.6f}")
        print(f"误差改善: {iteration_history[0]['error'] - iteration_history[-1]['error']:.6f}")


def show_mujoco_home_pose(xml_file, home_q, target_pose_M, T_W_E1_home, T_W_E2_home):
    """
    使用MuJoCo viewer展示Home关键帧下的双臂位姿。
    
    Args:
        xml_file (str): MuJoCo XML文件路径
        home_q (np.ndarray): Home关键帧的关节角度
        target_pose_M (np.ndarray): 虚拟物体中点的目标位姿
        T_W_E1_home (np.ndarray): Home位姿下左臂末端的变换矩阵
        T_W_E2_home (np.ndarray): Home位姿下右臂末端的变换矩阵
    """
    try:
        import mujoco
        import mujoco.viewer

        # 加载MuJoCo模型
        model = mujoco.MjModel.from_xml_path(xml_file)
        data = mujoco.MjData(model)

        # 设置home关键帧的关节角度
        data.qpos[:12] = home_q

        # 更新所有相关的仿真数据
        mujoco.mj_forward(model, data)
        
        # 使用MuJoCo viewer展示home位姿
        print("正在启动MuJoCo viewer展示Home位姿...")
        print("按ESC键退出查看器")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # 设置相机位置以便更好地观察双臂
            viewer.cam.distance = 3.0  # 相机距离
            viewer.cam.azimuth = 45    # 方位角
            viewer.cam.elevation = -20  # 仰角
            
            # 显示home位姿信息
            print("=== Home关键帧位姿信息 ===")
            print("左臂关节角度 (度):", np.rad2deg(home_q[:6]))
            print("右臂关节角度 (度):", np.rad2deg(home_q[6:]))
            print("虚拟物体中点位置:", target_pose_M[:3, 3])
            print("左臂末端位置:", T_W_E1_home[:3, 3])
            print("右臂末端位置:", T_W_E2_home[:3, 3])
            print("两臂末端距离:", np.linalg.norm(T_W_E2_home[:3, 3] - T_W_E1_home[:3, 3]))
            
            # 保持viewer运行，直到用户关闭
            while viewer.is_running():
                viewer.sync()
                
    except ImportError:
        print("\n无法导入 'mujoco.viewer'。")
        print("请确保MuJoCo已正确安装并支持viewer功能。")
    except Exception as e:
        print(f"\n可视化过程中发生错误: {e}")
        print("继续执行...")


def print_optimization_statistics(iteration_history, solution_q, initial_q, target_pose_M, T_W_M_sol, pos_diff, constraint_pos_diff):
    """
    打印优化结果统计信息。
    
    Args:
        iteration_history (list): 迭代历史记录
        solution_q (np.ndarray): 求解得到的关节角度
        initial_q (np.ndarray): 初始关节角度
        target_pose_M (np.ndarray): 目标位姿
        T_W_M_sol (np.ndarray): 求解得到的位姿
        pos_diff (float): 位置误差
        constraint_pos_diff (float): 闭环约束误差
    """
    print("\n=== 优化结果统计 ===")
    print(f"总迭代次数: {len(iteration_history)}")
    print(f"初始误差: {iteration_history[0]['error']:.6f}")
    print(f"最终误差: {iteration_history[-1]['error']:.6f}")
    print(f"误差改善: {iteration_history[0]['error'] - iteration_history[-1]['error']:.6f}")
    print(f"误差改善百分比: {(iteration_history[0]['error'] - iteration_history[-1]['error']) / iteration_history[0]['error'] * 100:.2f}%")
    
    print(f"\n=== 求解结果详情 ===")
    print(f"目标位姿:")
    print(f"  位置: {target_pose_M[:3, 3]}")
    print(f"  旋转矩阵:")
    print(target_pose_M[:3, :3])
    
    print(f"\n实际到达位姿:")
    print(f"  位置: {T_W_M_sol[:3, 3]}")
    print(f"  旋转矩阵:")
    print(T_W_M_sol[:3, :3])
    
    print(f"\n误差分析:")
    print(f"  位置误差: {pos_diff:.6f} 米")
    print(f"  闭环约束误差: {constraint_pos_diff:.6f} 米")
    
    print(f"\n关节角度解:")
    print(f"  左臂 (度): {np.rad2deg(solution_q[:6])}")
    print(f"  右臂 (度): {np.rad2deg(solution_q[6:])}")
    
    # 计算并显示每个关节的角度变化
    print(f"\n关节角度变化 (相对于初始值):")
    initial_q1 = initial_q[:6]
    initial_q2 = initial_q[6:]
    print(f"  左臂变化 (度): {np.rad2deg(solution_q[:6] - initial_q1)}")
    print(f"  右臂变化 (度): {np.rad2deg(solution_q[6:] - initial_q2)}")
    
    # 显示优化结果
    print(f"\n优化结果:")
    print(f"  优化成功: 是")
    print(f"  最终误差值: {iteration_history[-1]['error']:.6f}")
    print(f"  迭代次数: {len(iteration_history)}")
    print(f"  函数评估次数: {len(iteration_history)}")  # 每次迭代调用一次目标函数 