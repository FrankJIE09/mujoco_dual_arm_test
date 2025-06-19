"""
双臂机器人运动学求解器

这个模块实现了双臂机器人的闭环逆运动学求解。主要功能包括：
1. 从MuJoCo XML文件解析机器人模型
2. 创建ikpy运动学链
3. 求解双臂协同操作的逆运动学问题
4. 验证求解结果并生成可视化

数学原理：
- 使用齐次变换矩阵进行坐标变换
- 闭环约束：T_E1_E2 = T_E1_M @ inv(T_E2_M)
- 优化目标：minimize(位置误差 + 姿态误差 + λ×闭环约束误差)

作者: [您的姓名]
日期: [创建日期]
"""

import ikpy.chain
import ikpy.link
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize


def rpy_to_matrix(rpy):
    """
    将RPY欧拉角转换为旋转矩阵。

    数学公式：
    R = Rz(ψ) @ Ry(θ) @ Rx(φ)
    其中：
    - φ (roll): 绕X轴旋转
    - θ (pitch): 绕Y轴旋转
    - ψ (yaw): 绕Z轴旋转

    Args:
        rpy (list or np.ndarray): 包含roll, pitch, yaw的欧拉角 [rx, ry, rz]

    Returns:
        np.ndarray: 3x3旋转矩阵
    """
    return Rotation.from_euler('xyz', rpy, degrees=False).as_matrix()


def quat_to_matrix(quat):
    """
    将四元数转换为旋转矩阵。

    数学公式：
    q = [x, y, z, w] = [q1, q2, q3, q0]
    R = [1-2q2²-2q3²    2(q1q2-q0q3)    2(q1q3+q0q2)]
        [2(q1q2+q0q3)   1-2q1²-2q3²     2(q2q3-q0q1)]
        [2(q1q3-q0q2)   2(q2q3+q0q1)    1-2q1²-2q2²]

    Args:
        quat (list or np.ndarray): 四元数 [x, y, z, w]

    Returns:
        np.ndarray: 3x3旋转矩阵
    """
    return Rotation.from_quat(quat).as_matrix()


def pose_to_transformation_matrix(pose):
    """
    将位姿向量转换为4x4变换矩阵。

    数学公式：
    pose = [x, y, z, qx, qy, qz, qw]
    T = [R    p] 其中 R = quat_to_matrix([qx, qy, qz, qw])
        [0    1]      p = [x, y, z]ᵀ

    Args:
        pose (list or np.ndarray): 位姿向量 [x, y, z, qx, qy, qz, qw]

    Returns:
        np.ndarray: 4x4齐次变换矩阵
    """
    t = np.eye(4)
    t[:3, 3] = pose[:3]  # 设置平移部分 p = [x, y, z]ᵀ
    t[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()  # 设置旋转部分 R
    return t


def transformation_matrix_to_pose(t):
    """
    将4x4变换矩阵转换为位姿向量。

    数学公式：
    T = [R    p] → pose = [p_x, p_y, p_z, q_x, q_y, q_z, q_w]
        [0    1]
    其中 q = matrix_to_quat(R)

    Args:
        t (np.ndarray): 4x4齐次变换矩阵

    Returns:
        np.ndarray: 位姿向量 [x, y, z, qx, qy, qz, qw]
    """
    pose = np.zeros(7)
    pose[:3] = t[:3, 3]  # 提取平移部分 p
    pose[3:] = Rotation.from_matrix(t[:3, :3]).as_quat()  # 提取旋转部分（四元数）
    return pose


def get_transformation(body_element):
    """
    从MuJoCo的body XML元素中提取其相对于父坐标系的4x4变换矩阵。

    这个函数解析XML中的位置和姿态信息，支持以下属性：
    - pos: 3D位置向量
    - quat: 四元数姿态 (MuJoCo格式: w,x,y,z)
    - euler: 欧拉角姿态 (假设xyz顺序)

    数学公式：
    T = [R    p] 其中 p = pos, R = quat_to_matrix(quat) 或 euler_to_matrix(euler)
        [0    1]

    Args:
        body_element (xml.etree.ElementTree.Element): MuJoCo XML中的body元素

    Returns:
        np.ndarray: 4x4齐次变换矩阵
    """
    # 解析位置信息
    pos = body_element.get('pos')
    pos = np.fromstring(pos, sep=' ') if pos else np.zeros(3)

    # 解析姿态信息 (优先使用四元数)
    quat = body_element.get('quat')
    euler = body_element.get('euler')

    rot_matrix = np.eye(3)
    if quat:
        # MuJoCo的四元数顺序是 (w, x, y, z), 而Scipy需要 (x, y, z, w)
        # 转换公式: [w, x, y, z] → [x, y, z, w]
        q = np.fromstring(quat, sep=' ')
        rot_matrix = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    elif euler:
        # 假设欧拉角顺序为 'xyz'
        # 旋转矩阵: R = Rz(ψ) @ Ry(θ) @ Rx(φ)
        e = np.fromstring(euler, sep=' ')
        rot_matrix = Rotation.from_euler('XYZ', e).as_matrix()

    # 构建4x4变换矩阵
    # 数学公式: T = [R    p]
    #                [0    1]
    transformation = np.eye(4)
    transformation[:3, :3] = rot_matrix  # 旋转部分 R
    transformation[:3, 3] = pos  # 平移部分 p
    return transformation


def create_chain_from_mjcf(xml_file, base_body_name):
    """
    解析MuJoCo XML文件，为指定的机械臂创建ikpy运动学链。

    这个函数遍历XML中的连杆结构，提取关节信息，并创建ikpy的Chain对象。
    支持6自由度机械臂，每个关节都有位置和姿态限制。

    运动学链结构：
    base → link1 → link2 → ... → link6 → end_effector
    每个连杆的变换: T_i = [R_i    p_i]
                          [0      1  ]

    Args:
        xml_file (str): MuJoCo XML文件的路径
        base_body_name (str): 机械臂基座的body名称 (例如, 'left_robot_base')

    Returns:
        tuple: (ikpy.chain.Chain, np.ndarray)
               - 创建的运动学链
               - 基座在世界坐标系下的变换矩阵
    """
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()
    worldbody = root.find('worldbody')

    # 找到机械臂的基座元素并获取其在世界坐标系下的变换
    base_element = worldbody.find(f".//body[@name='{base_body_name}']")
    base_transform = get_transformation(base_element)

    # 创建ikpy链的连杆列表
    # 第一个连杆通常是固定的世界坐标系或基座
    links = [ikpy.link.URDFLink(
        name="base",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0]
    )]

    # active_links_mask 用于告诉ikpy哪些关节是可动的
    # 第一个基座连杆是固定的 (False)
    active_links_mask = [False]

    # 从基座开始，迭代遍历6个连杆
    current_element = base_element
    for i in range(1, 7):
        # 构建连杆名称 (例如: left_link1, left_link2, ...)
        link_name = f"{base_body_name.split('_')[0]}_link{i}"
        current_element = current_element.find(f".//body[@name='{link_name}']")
        if current_element is None:
            break

        # 从XML中提取关节信息
        joint_element = current_element.find('joint')
        joint_name = joint_element.get('name')
        joint_axis = np.fromstring(joint_element.get('axis'), sep=' ')
        joint_range = joint_element.get('range')
        bounds = tuple(map(float, joint_range.split())) if joint_range else (None, None)

        # 获取连杆相对于其父连杆的变换
        link_transform = get_transformation(current_element)
        translation = link_transform[:3, 3]  # 平移部分
        orientation_matrix = link_transform[:3, :3]  # 旋转部分
        # 将旋转矩阵转换为ikpy期望的RPY欧拉角格式
        # 数学公式: [φ, θ, ψ] = matrix_to_euler(R, 'xyz')
        orientation_rpy = Rotation.from_matrix(orientation_matrix).as_euler('xyz')

        # 创建ikpy连杆对象
        link = ikpy.link.URDFLink(
            name=joint_name,
            origin_translation=translation,  # 连杆原点相对于父连杆的平移
            origin_orientation=orientation_rpy,  # 连杆原点相对于父连杆的旋转(RPY)
            rotation=joint_axis,  # 关节的旋转轴
            bounds=bounds  # 关节角度限制
        )
        links.append(link)
        active_links_mask.append(True)  # 这6个关节都是可动的

    # 添加末端执行器标记 (如果有的话)，它是一个固定连杆
    ee_body = None
    for body in current_element.iter('body'):
        if '_end_effector' in body.get('name', ''):
            ee_body = body
            break

    if ee_body is not None:
        ee_transform = get_transformation(ee_body)
        ee_orientation_matrix = ee_transform[:3, :3]
        ee_orientation_rpy = Rotation.from_matrix(ee_orientation_matrix).as_euler('xyz')
        ee_link = ikpy.link.URDFLink(
            name=ee_body.get('name'),
            origin_translation=ee_transform[:3, 3],
            origin_orientation=ee_orientation_rpy,
            rotation=[0, 0, 0]  # 固定连杆，无旋转
        )
        links.append(ee_link)
        active_links_mask.append(False)  # 末端执行器标记是固定的

    # 使用创建的连杆列表和活动关节掩码来构建运动学链
    chain = ikpy.chain.Chain(links, active_links_mask=active_links_mask)
    return chain, base_transform


def get_kinematics(xml_file_path):
    """
    为双臂机器人创建运动学模型。

    这个函数为左右两个机械臂分别创建运动学链，并返回它们的基座变换矩阵。

    Args:
        xml_file_path (str): MuJoCo XML文件的路径

    Returns:
        tuple: (左臂链, 右臂链, 左臂基座变换, 右臂基座变换)
    """
    left_chain, left_base_transform = create_chain_from_mjcf(xml_file_path, 'left_robot_base')
    right_chain, right_base_transform = create_chain_from_mjcf(xml_file_path, 'right_robot_base')
    return left_chain, right_chain, left_base_transform, right_base_transform


def solve_dual_arm_ik(target_pose_M, initial_q, chains_and_bases):
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

    Returns:
        np.ndarray or None: 求解得到的12个关节角度，如果求解失败则返回None
    """
    left_chain, right_chain, T_W_B1, T_W_B2 = chains_and_bases

    # --- 定义闭环约束 ---
    # 假设虚拟物体是一个20cm宽的杆，两个夹爪各抓住一端。
    # M是杆的中点，所以每个夹爪距离M为10cm。

    # T_E1_M: 从左臂末端(E1)坐标系到中点(M)坐标系的变换。
    # 假设E1的Z轴朝向M，所以M在E1的Z轴正方向0.24m处。
    # 数学公式: T_E1_M = [I    [0, 0, 0.24]ᵀ]
    #                     [0    1           ]
    T_E1_M = np.eye(4)
    T_E1_M[2, 3] = 0.24

    # T_E2_M: 从右臂末端(E2)坐标系到中点(M)坐标系的变换。
    # 为了让E2与E1相对，E2需要绕其Y轴旋转180度。
    # 旋转后，M也位于E2的(新)Z轴正方向0.24m处。
    # 数学公式: T_E2_M = R_y(π) @ [I    [0, 0, 0.24]ᵀ]
    #                               [0    1            ]
    # 其中 R_y(π) = [cos(π)  0  sin(π)] = [-1  0  0]
    #                [0       1  0     ]   [0   1  0]
    #                [-sin(π) 0  cos(π)]   [0   0  -1]
    T_E2_M = np.eye(4)
    T_E2_M[:3, :3] = Rotation.from_euler('y', 180, degrees=True).as_matrix()
    T_E2_M[:3, 3] = np.array([0, 0, 0.24])

    # T_E1_E2 是从左臂末端到右臂末端的恒定变换，这是核心的闭环约束。
    # 数学公式: T_E1_E2 = T_E1_M @ inv(T_E2_M)
    # 这个变换必须保持恒定，确保两个手臂末端始终保持正确的相对位姿
    T_E1_E2 = T_E1_M @ np.linalg.inv(T_E2_M)

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
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-4}  # 优化器选项
    )

    if result.success:
        print("IK solution found.")
        return result.x  # 返回找到的关节角度
    else:
        print("IK solution not found.")
        print(result.message)
        return None


if __name__ == '__main__':
    """
    主程序：演示双臂运动学求解的完整流程

    包括以下步骤：
    1. 解析XML文件并创建运动学链
    2. 定义目标位姿
    3. 求解逆运动学
    4. 验证结果
    5. 生成可视化
    """
    xml_file = 'dual_elfin15_scene_clean.xml'

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
            viewer.cam.azimuth = 45  # 方位角
            viewer.cam.elevation = -20  # 仰角

            # 显示home位姿信息
            print("=== Home关键帧位姿信息 ===")
            print("左臂关节角度 (度):", np.rad2deg(home_q1))
            print("右臂关节角度 (度):", np.rad2deg(home_q2))
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
        print("继续执行逆运动学求解...")

    # --- 3. 求解阶段 ---
    # 使用home关键帧的关节角度作为优化的初始猜测值
    initial_q = home_q.copy()

    print(f"\n使用'home'关键帧的关节角度作为初始猜测值: {initial_q}")

    print("\n正在求解双臂逆运动学...")
    solution_q = solve_dual_arm_ik(target_pose_M, initial_q, kinematics_data)

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

        # --- 5. 可视化阶段 ---
        print("\n正在使用MuJoCo viewer展示结果...")
        try:
            import mujoco
            import mujoco.viewer

            # 加载MuJoCo模型
            model = mujoco.MjModel.from_xml_path(xml_file)
            data = mujoco.MjData(model)

            # 将计算出的关节角度设置为模型的姿态
            data.qpos[:12] = solution_q

            # 更新所有相关的仿真数据 (例如几何体的位置)
            mujoco.mj_forward(model, data)

            # 使用MuJoCo viewer直接展示结果
            print("正在启动MuJoCo viewer...")
            print("按ESC键退出查看器")

            with mujoco.viewer.launch_passive(model, data) as viewer:
                # 设置相机位置以便更好地观察双臂
                viewer.cam.distance = 3.0  # 相机距离
                viewer.cam.azimuth = 45  # 方位角
                viewer.cam.elevation = -20  # 仰角

                # 显示求解结果
                print("双臂机器人已到达目标位姿")
                print("左臂关节角度 (度):", np.rad2deg(q1_sol))
                print("右臂关节角度 (度):", np.rad2deg(q2_sol))
                print("虚拟物体中点位置:", T_W_M_sol[:3, 3])
                print("位置误差:", f"{pos_diff:.6f} 米")
                print("闭环约束误差:", f"{constraint_pos_diff:.6f} 米")

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
            print(f"  左臂 (度): {np.rad2deg(q1_sol)}")
            print(f"  右臂 (度): {np.rad2deg(q2_sol)}")

            # 计算并显示每个关节的角度变化
            print(f"\n关节角度变化 (相对于初始值):")
            initial_q1 = initial_q[:6]
            initial_q2 = initial_q[6:]
            print(f"  左臂变化 (度): {np.rad2deg(q1_sol - initial_q1)}")
            print(f"  右臂变化 (度): {np.rad2deg(q2_sol - initial_q2)}")

            # 显示优化结果
            print(f"\n优化结果:")
            print(f"  优化成功: 是")
            print(f"  最终误差值: {result.fun:.6f}")
            print(f"  迭代次数: {result.nit}")
            print(f"  函数评估次数: {result.nfev}")
