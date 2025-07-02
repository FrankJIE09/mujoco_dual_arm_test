"""
逆运动学求解模块

这个模块实现了双臂系统的闭环逆运动学求解算法。
主要功能包括：
1. 定义闭环约束和目标函数
2. 使用优化方法求解逆运动学
3. 记录迭代历史
4. 验证求解结果

数学原理：
闭环约束: T_E1_E2 = T_E1_M @ inv(T_E2_M) = constant
优化目标: minimize(||p_M - p_target|| + α||R_M - R_target|| + λ||T_E1_E2 - T_E1_E2_target||)

作者: frank
日期: 2024年6月19日
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize


def solve_dual_arm_ik(target_pose_M, initial_q, chains_and_bases, T_E1_M=None, T_E2_M=None):
    """
    求解双臂系统的闭环逆运动学。
    
    这个函数使用优化方法求解双臂协同操作的逆运动学问题。核心思想是：
    1. 两个机械臂末端通过虚拟物体连接，形成闭环约束
    2. 虚拟物体的中点M有目标位姿要求
    3. 通过优化关节角度，同时满足目标位姿和闭环约束
    
    数学原理：
    闭环约束: T_E1_E2 = T_E1_M @ inv(T_E2_M) = constant
    优化目标: minimize(||p_M - p_target|| + α||R_M - R_target|| + λ||T_E1_E2 - T_E1_E2_target||)
    
    Args:
        target_pose_M (np.ndarray): 虚拟物体中点M的目标4x4变换矩阵
        initial_q (np.ndarray): 12个关节角度的初始猜测值 (左臂6个 + 右臂6个)
        chains_and_bases (tuple): 包含运动学链和基座变换的元组
        T_E1_M (np.ndarray, optional): 左臂末端到虚拟物体中心的4x4变换矩阵。
                                      如果为None，使用默认值（无偏移，M与E1重合）
        T_E2_M (np.ndarray, optional): 右臂末端到虚拟物体中心的4x4变换矩阵。
                                      如果为None，使用默认值（Y轴旋转180度，Z轴偏移0.40m）
    
    Returns:
        tuple: (求解得到的12个关节角度, 迭代历史记录) 如果求解失败则返回(None, None)
    """
    left_chain, right_chain, T_W_B1, T_W_B2 = chains_and_bases

    # --- 定义闭环约束 ---
    # 使用用户提供的变换矩阵，或者使用默认值

    # T_E1_M: 从左臂末端(E1)坐标系到中点(M)坐标系的变换
    if T_E1_M is None:
        # 默认值：M与E1重合（无偏移）
        T_E1_M = np.eye(4)
        T_E1_M[2, 3] = 0.0
    else:
        T_E1_M = np.array(T_E1_M, dtype=float)

    # T_E2_M: 从右臂末端(E2)坐标系到中点(M)坐标系的变换
    if T_E2_M is None:
        # 默认值：为了让E2与E1相对，E2需要绕其Y轴旋转180度
        # 旋转后，M位于E2的Z轴正方向0.40m处
        T_E2_M = np.eye(4)
        T_E2_M[:3, :3] = Rotation.from_euler('y', 180, degrees=True).as_matrix()
        T_E2_M[:3, 3] = np.array([0, 0, 0.40])
    else:
        T_E2_M = np.array(T_E2_M, dtype=float)

    # T_E1_E2 是从左臂末端到右臂末端的恒定变换，这是核心的闭环约束。
    # 数学公式: T_E1_E2 = T_E1_M @ inv(T_E2_M)
    # 这个变换必须保持恒定，确保两个手臂末端始终保持正确的相对位姿
    T_E1_E2 = T_E1_M @ np.linalg.inv(T_E2_M)

    # 用于记录迭代历史的列表
    iteration_history = []

    def objective(q):
        """
        优化问题的目标函数。

        这个函数计算当前关节角度配置下的误差，包括：
        1. 任务空间误差：虚拟物体中点M的当前位姿与目标位姿的差距
        2. 闭环约束误差：两个手臂末端当前的相对位姿与预设的恒定位姿的差距
        
        数学公式：
        error = ||p_M - p_target|| + α(1 - |q_M · q_target|) + λ(||p_E1_E2 - p_E1_E2_target|| + β(1 - |q_E1_E2 · q_E1_E2_target|))
        
        其中：
        - p_M: 虚拟物体中点的当前位置
        - p_target: 虚拟物体中点的目标位置
        - q_M: 虚拟物体中点的当前姿态四元数
        - q_target: 虚拟物体中点的目标姿态四元数
        - p_E1_E2: 两臂末端的当前相对位置
        - p_E1_E2_target: 两臂末端的目标相对位置
        - q_E1_E2: 两臂末端的当前相对姿态四元数
        - q_E1_E2_target: 两臂末端的目标相对姿态四元数
        - α, β: 姿态误差权重
        - λ: 闭环约束权重（通常很大，确保约束严格满足）
        
        Args:
            q (np.ndarray): 12个关节角度的向量 (左臂6个 + 右臂6个)
        
        Returns:
            float: 总误差值，越小越好
        """
        q1 = q[:6]  # 左臂6个关节角度
        q2 = q[6:]  # 右臂6个关节角度

        # ikpy的正向运动学需要一个包含所有连杆(包括固定连杆)的角度向量
        # 所以我们需要在首尾填充0
        fk_q1 = np.concatenate(([0], q1, [0]))
        fk_q2 = np.concatenate(([0], q2, [0]))

        # --- 正向运动学计算 ---
        # 数学公式: T_B1_E1 = ∏(i=1 to 6) T_i(q_i)
        # T_B1_E1: 左臂基座到末端的变换
        T_B1_E1 = left_chain.forward_kinematics(fk_q1)
        # T_B2_E2: 右臂基座到末端的变换
        T_B2_E2 = right_chain.forward_kinematics(fk_q2)

        # 世界坐标系下的末端位姿
        # 数学公式: T_W_E1 = T_W_B1 @ T_B1_E1
        # T_W_E1: 世界坐标系到左臂末端的变换
        T_W_E1 = T_W_B1 @ T_B1_E1
        # T_W_E2: 世界坐标系到右臂末端的变换
        T_W_E2 = T_W_B2 @ T_B2_E2

        # --- 计算误差 ---
        # 1. 任务空间误差: 虚拟物体中点M的当前位姿与目标位姿的差距
        # M的当前位姿可以由任一手臂计算得出，这里用左臂
        # 数学公式: T_W_M = T_W_E1 @ T_E1_M
        T_W_M_from_E1 = T_W_E1 @ T_E1_M
        
        # 位置误差: 计算两个点之间的欧氏距离
        # 数学公式: pos_error = ||p_M - p_target||
        pos_error = np.linalg.norm(T_W_M_from_E1[:3, 3] - target_pose_M[:3, 3])
        
        # 姿态误差: 计算两个姿态四元数之间的点积。点积绝对值越接近1，姿态越接近。
        # 数学公式: orient_error = 1 - |q_M · q_target|
        orient_error_quat = Rotation.from_matrix(T_W_M_from_E1[:3, :3]).as_quat()
        target_orient_quat = Rotation.from_matrix(target_pose_M[:3, :3]).as_quat()
        orient_error = 1.0 - np.abs(np.dot(orient_error_quat, target_orient_quat))

        # 2. 闭环约束误差: 两个手臂末端当前的相对位姿与预设的恒定位姿的差距
        # 数学公式: T_E1_E2_current = inv(T_W_E1) @ T_W_E2
        T_E1_E2_current = np.linalg.inv(T_W_E1) @ T_W_E2
        
        # 约束的位置误差
        # 数学公式: constraint_pos_error = ||p_E1_E2_current - p_E1_E2_target||
        constraint_pos_error = np.linalg.norm(T_E1_E2_current[:3, 3] - T_E1_E2[:3, 3])
        
        # 约束的姿态误差
        # 数学公式: constraint_orient_error = 1 - |q_E1_E2_current · q_E1_E2_target|
        constraint_rot_current = Rotation.from_matrix(T_E1_E2_current[:3, :3]).as_quat()
        constraint_rot_target = Rotation.from_matrix(T_E1_E2[:3, :3]).as_quat()
        constraint_orient_error = 1.0 - np.abs(np.dot(constraint_rot_current, constraint_rot_target))

        # --- 组合总误差 ---
        # 最终的误差是任务空间误差和闭环约束误差的加权和。
        # 我们给闭环约束误差一个很大的权重(100)，因为这个约束必须被严格遵守。
        # 数学公式: total_error = pos_error + orient_error + λ(constraint_pos_error + constraint_orient_error)
        error = pos_error + orient_error + 100 * (constraint_pos_error + constraint_orient_error)
        return error

    def callback(xk):
        """
        优化器的回调函数，用于记录每次迭代的信息
        """
        # 记录当前迭代的关节角度和误差
        current_error = objective(xk)
        iteration_info = {
            'iteration': len(iteration_history) + 1,
            'joint_angles': xk.copy(),
            'error': current_error,
            'left_arm_angles': xk[:6],
            'right_arm_angles': xk[6:]
        }
        iteration_history.append(iteration_info)
        
        # 每10次迭代打印一次进度
        if len(iteration_history) % 10 == 0:
            print(f"迭代 {len(iteration_history)}: 误差 = {current_error:.6f}")

    # 从运动学链中为活动关节提取关节限制 (bounds)
    bounds_l = [l.bounds for l, active in zip(left_chain.links, left_chain.active_links_mask) if active]
    bounds_r = [l.bounds for l, active in zip(right_chain.links, right_chain.active_links_mask) if active]
    bounds = bounds_l + bounds_r

    # 定义约束条件
    def constraint_closed_loop(q):
        """
        闭环约束：确保两臂末端的相对位姿满足预设的恒定位姿
        约束形式：||T_E1_E2_current - T_E1_E2_target|| = 0
        
        Args:
            q (np.ndarray): 12个关节角度的向量
        
        Returns:
            float: 约束违反程度，0表示满足约束
        """
        q1 = q[:6]  # 左臂6个关节角度
        q2 = q[6:]  # 右臂6个关节角度

        # ikpy正向运动学
        fk_q1 = np.concatenate(([0], q1, [0]))
        fk_q2 = np.concatenate(([0], q2, [0]))

        T_B1_E1 = left_chain.forward_kinematics(fk_q1)
        T_B2_E2 = right_chain.forward_kinematics(fk_q2)

        T_W_E1 = T_W_B1 @ T_B1_E1
        T_W_E2 = T_W_B2 @ T_B2_E2

        # 计算当前两臂末端的相对位姿
        T_E1_E2_current = np.linalg.inv(T_W_E1) @ T_W_E2
        
        # 计算与目标相对位姿的差距
        pos_diff = np.linalg.norm(T_E1_E2_current[:3, 3] - T_E1_E2[:3, 3])
        rot_diff = 1.0 - np.abs(np.dot(
            Rotation.from_matrix(T_E1_E2_current[:3, :3]).as_quat(),
            Rotation.from_matrix(T_E1_E2[:3, :3]).as_quat()
        ))
        
        return pos_diff + rot_diff

    def constraint_position_tolerance(q):
        """
        位置容差约束：确保虚拟物体中点M的位置误差在可接受范围内
        约束形式：||p_M - p_target|| <= tolerance
        
        Args:
            q (np.ndarray): 12个关节角度的向量
        
        Returns:
            float: 位置误差减去容差，负值表示满足约束
        """
        q1 = q[:6]
        q2 = q[6:]

        fk_q1 = np.concatenate(([0], q1, [0]))
        fk_q2 = np.concatenate(([0], q2, [0]))

        T_B1_E1 = left_chain.forward_kinematics(fk_q1)
        T_W_E1 = T_W_B1 @ T_B1_E1
        T_W_M_from_E1 = T_W_E1 @ T_E1_M
        
        pos_error = np.linalg.norm(T_W_M_from_E1[:3, 3] - target_pose_M[:3, 3])
        tolerance = 0.05  # 5cm容差
        
        return pos_error - tolerance

    # 定义约束条件列表
    constraints = [
        {
            'type': 'eq',
            'fun': constraint_closed_loop,
            'name': 'closed_loop_constraint'
        },
        {
            'type': 'ineq',
            'fun': lambda q: -constraint_position_tolerance(q),  # 转换为 <= 0 形式
            'name': 'position_tolerance_constraint'
        }
    ]

    # 使用Scipy的minimize函数进行优化
    # 优化问题: minimize objective(q) subject to bounds and constraints
    result = minimize(
        objective,  # 目标函数 (我们要最小化的)
        initial_q,  # 初始猜测值
        method='SLSQP',  # 一种支持约束和边界的优化算法
        bounds=bounds,  # 关节角度的边界
        constraints=constraints,  # 约束条件
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-4},  # 优化器选项
        callback=callback  # 回调函数，记录迭代历史
    )

    if result.success:
        print("IK solution found.")
        return result.x, iteration_history  # 返回找到的关节角度和迭代历史
    else:
        print("IK solution not found.")
        print(result.message)
        return None, iteration_history 


def create_transform_matrix(position=None, rotation=None, rotation_type='euler', rotation_unit='degrees'):
    """
    创建4x4变换矩阵的辅助函数
    
    Args:
        position (array-like, optional): [x, y, z] 位置向量，默认为 [0, 0, 0]
        rotation (array-like, optional): 旋转参数，默认为无旋转
        rotation_type (str): 旋转参数类型，可选：
                           - 'euler': 欧拉角 [roll, pitch, yaw]
                           - 'quat': 四元数 [x, y, z, w]
                           - 'rotvec': 旋转向量 [x, y, z]
                           - 'matrix': 3x3旋转矩阵
        rotation_unit (str): 欧拉角和旋转向量的单位，'degrees' 或 'radians'
    
    Returns:
        np.ndarray: 4x4变换矩阵
    
    Examples:
        # 创建一个仅有位移的变换矩阵
        T = create_transform_matrix(position=[0.1, 0.2, 0.3])
        
        # 创建一个绕Y轴旋转180度的变换矩阵
        T = create_transform_matrix(rotation=[0, 180, 0], rotation_type='euler')
        
        # 创建一个同时有位移和旋转的变换矩阵
        T = create_transform_matrix(position=[0, 0, 0.4], 
                                  rotation=[0, 180, 0], 
                                  rotation_type='euler')
    """
    T = np.eye(4)
    
    # 设置位置
    if position is not None:
        T[:3, 3] = np.array(position)
    
    # 设置旋转
    if rotation is not None:
        rotation = np.array(rotation)
        
        if rotation_type == 'euler':
            # 欧拉角（XYZ顺序）
            if rotation_unit == 'degrees':
                R = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
            else:
                R = Rotation.from_euler('xyz', rotation, degrees=False).as_matrix()
        elif rotation_type == 'quat':
            # 四元数 [x, y, z, w]
            R = Rotation.from_quat(rotation).as_matrix()
        elif rotation_type == 'rotvec':
            # 旋转向量
            if rotation_unit == 'degrees':
                rotation = np.deg2rad(rotation)
            R = Rotation.from_rotvec(rotation).as_matrix()
        elif rotation_type == 'matrix':
            # 旋转矩阵
            R = np.array(rotation)
        else:
            raise ValueError(f"未知的旋转类型: {rotation_type}")
        
        T[:3, :3] = R
    
    return T


def create_default_transforms():
    """
    创建默认的变换矩阵
    
    Returns:
        tuple: (T_E1_M, T_E2_M) 默认的左臂和右臂到虚拟物体中心的变换矩阵
    """
    # 左臂末端到虚拟物体中心：无偏移
    T_E1_M = create_transform_matrix(position=[0, 0, 0])
    
    # 右臂末端到虚拟物体中心：Y轴旋转180度，Z轴偏移0.40m
    T_E2_M = create_transform_matrix(position=[0, 0, 0.40], 
                                    rotation=[0, 180, 0], 
                                    rotation_type='euler')
    
    return T_E1_M, T_E2_M


def create_grasp_transforms(object_width, grasp_offset=0.0):
    """
    为抓取特定宽度物体创建变换矩阵
    
    Args:
        object_width (float): 物体宽度（米）
        grasp_offset (float): 手爪相对于物体边缘的额外偏移（米）
    
    Returns:
        tuple: (T_E1_M, T_E2_M) 适用于抓取指定宽度物体的变换矩阵
    
    Examples:
        # 抓取20cm宽的物体
        T_E1_M, T_E2_M = create_grasp_transforms(0.20)
        
        # 抓取30cm宽的物体，手爪额外内收5cm
        T_E1_M, T_E2_M = create_grasp_transforms(0.30, grasp_offset=-0.05)
    """
    half_width = object_width / 2.0 + grasp_offset
    
    # 左臂末端到物体中心：沿Z轴负方向偏移
    T_E1_M = create_transform_matrix(position=[0, 0, -half_width])
    
    # 右臂末端到物体中心：Y轴旋转180度，沿Z轴正方向偏移
    T_E2_M = create_transform_matrix(position=[0, 0, half_width], 
                                    rotation=[0, 180, 0], 
                                    rotation_type='euler')
    
    return T_E1_M, T_E2_M


def print_transform_info(T, name="变换矩阵"):
    """
    打印变换矩阵的详细信息
    
    Args:
        T (np.ndarray): 4x4变换矩阵
        name (str): 矩阵名称
    """
    print(f"\n=== {name} ===")
    print(f"位置 (x, y, z): [{T[0,3]:.6f}, {T[1,3]:.6f}, {T[2,3]:.6f}]")
    
    # 提取欧拉角
    rotation = Rotation.from_matrix(T[:3, :3])
    euler_xyz = rotation.as_euler('xyz', degrees=True)
    print(f"欧拉角 XYZ (度): [{euler_xyz[0]:.2f}, {euler_xyz[1]:.2f}, {euler_xyz[2]:.2f}]")
    
    # 提取四元数
    quat = rotation.as_quat()
    print(f"四元数 (x,y,z,w): [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
    
    print("完整变换矩阵:")
    print(T)
    print("-" * 50) 