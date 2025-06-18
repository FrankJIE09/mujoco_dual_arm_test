import ikpy.chain
import ikpy.link
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

def rpy_to_matrix(rpy):
    """(此函数未使用) 将RPY欧拉角转换为旋转矩阵。"""
    return Rotation.from_euler('xyz', rpy, degrees=False).as_matrix()

def quat_to_matrix(quat):
    """(此函数未使用) 将四元数转换为旋转矩阵。"""
    return Rotation.from_quat(quat).as_matrix()

def pose_to_transformation_matrix(pose):
    """(此函数未使用) 将位姿 [x, y, z, qx, qy, qz, qw] 转换为4x4变换矩阵。"""
    t = np.eye(4)
    t[:3, 3] = pose[:3]
    t[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    return t

def transformation_matrix_to_pose(t):
    """(此函数未使用) 将4x4变换矩阵转换为位姿向量 [x, y, z, qx, qy, qz, qw]。"""
    pose = np.zeros(7)
    pose[:3] = t[:3, 3]
    pose[3:] = Rotation.from_matrix(t[:3, :3]).as_quat()
    return pose

def get_transformation(body_element):
    """
    从MuJoCo的body XML元素中提取其相对于父坐标系的4x4变换矩阵。
    它会解析 'pos' (位置), 'quat' (四元数) 或 'euler' (欧拉角) 属性。
    """
    # 解析位置
    pos = body_element.get('pos')
    pos = np.fromstring(pos, sep=' ') if pos else np.zeros(3)

    # 解析姿态 (四元数优先)
    quat = body_element.get('quat')
    euler = body_element.get('euler')

    rot_matrix = np.eye(3)
    if quat:
        # MuJoCo的四元数顺序是 (w, x, y, z), 而Scipy需要 (x, y, z, w)
        q = np.fromstring(quat, sep=' ')
        rot_matrix = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    elif euler:
        # 假设欧拉角顺序为 'xyz'
        e = np.fromstring(euler, sep=' ')
        rot_matrix = Rotation.from_euler('xyz', e).as_matrix()
    
    # 构建4x4变换矩阵
    transformation = np.eye(4)
    transformation[:3, :3] = rot_matrix
    transformation[:3, 3] = pos
    return transformation

def create_chain_from_mjcf(xml_file, base_body_name):
    """
    解析一个MuJoCo XML文件，并为指定的机械臂创建一个ikpy运动学链 (Chain)。

    Args:
        xml_file (str): XML文件的路径。
        base_body_name (str): 机械臂基座的body名称 (例如, 'left_robot_base')。

    Returns:
        tuple: (ikpy.chain.Chain, np.ndarray) 包含创建的运动学链和基座在世界坐标系下的变换矩阵。
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    worldbody = root.find('worldbody')

    # 找到机械臂的基座元素并获取其在世界坐标系下的变换
    base_element = worldbody.find(f".//body[@name='{base_body_name}']")
    base_transform = get_transformation(base_element)

    # ikpy链的第一个连杆通常是固定的世界坐标系或基座
    links = [ikpy.link.URDFLink(name="base", origin_translation=[0,0,0], origin_orientation=[0,0,0], rotation=[0,0,0])]
    
    # active_links_mask 用于告诉ikpy哪些关节是可动的
    # 第一个基座连杆是固定的 (False)
    active_links_mask = [False]
    
    # 从基座开始，迭代遍历6个连杆
    current_element = base_element
    for i in range(1, 7):
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

        # 连杆的变换是相对于其父连杆的
        link_transform = get_transformation(current_element)
        translation = link_transform[:3, 3]
        orientation_matrix = link_transform[:3, :3]
        # 将旋转矩阵转换为ikpy期望的RPY欧拉角格式
        orientation_rpy = Rotation.from_matrix(orientation_matrix).as_euler('xyz')
        
        # 创建ikpy连杆对象
        link = ikpy.link.URDFLink(
            name=joint_name,
            origin_translation=translation,
            origin_orientation=orientation_rpy, # 使用RPY欧拉角
            rotation=joint_axis, # 关节的旋转轴
            bounds=bounds # 关节限制
        )
        links.append(link)
        active_links_mask.append(True) # 这6个关节都是可动的

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
            rotation=[0,0,0] # 固定连杆，无旋转
        )
        links.append(ee_link)
        active_links_mask.append(False) # 末端执行器标记是固定的


    # 使用创建的连杆列表和活动关节掩码来构建运动学链
    chain = ikpy.chain.Chain(links, active_links_mask=active_links_mask)
    return chain, base_transform

def get_kinematics(xml_file_path):
    """
    为双臂机器人创建运动学模型。

    Returns:
        tuple: (左臂链, 右臂链, 左臂基座变换, 右臂基座变换)
    """
    left_chain, left_base_transform = create_chain_from_mjcf(xml_file_path, 'left_robot_base')
    right_chain, right_base_transform = create_chain_from_mjcf(xml_file_path, 'right_robot_base')
    return left_chain, right_chain, left_base_transform, right_base_transform

def solve_dual_arm_ik(target_pose_M, initial_q, chains_and_bases):
    """
    求解双臂系统的闭环逆运动学。

    Args:
        target_pose_M (np.ndarray): 虚拟物体中点M的目标4x4变换矩阵。
        initial_q (np.ndarray): 12个关节角度的初始猜测值。
        chains_and_bases (tuple): 包含运动学链和基座变换的元组。
    """
    left_chain, right_chain, T_W_B1, T_W_B2 = chains_and_bases

    # --- 定义闭环约束 ---
    # 假设虚拟物体中点M位于左臂末端执行器(E1)的Z轴正方向0.1m处。
    # 这是 T_E1_M, 从E1坐标系到M坐标系的变换。
    T_E1_M = np.eye(4)
    T_E1_M[2, 3] = 0.1

    # 假设M也位于右臂末端执行器(E2)的Z轴正方向0.1m处。
    # 同时，为了让两个夹爪"相对"，我们将E2坐标系绕其X轴旋转180度。
    T_E2_M = np.eye(4)
    T_E2_M[2, 3] = 0.1 
    T_E2_M[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_matrix()

    # T_E1_E2 是从左臂末端到右臂末端的恒定变换，这是核心的闭环约束。
    # T_E1_E2 = T_E1_M @ T_M_E2 = T_E1_M @ inv(T_E2_M)
    T_M_E2 = np.linalg.inv(T_E2_M)
    T_E1_E2 = T_E1_M @ T_M_E2


    def objective(q):
        """
        优化问题的目标函数。
        我们的目标是最小化这个函数的返回值。
        q是包含12个关节角度的向量。
        """
        q1 = q[:6] # 左臂6个关节角度
        q2 = q[6:] # 右臂6个关节角度

        # ikpy的正向运动学需要一个包含所有连杆(包括固定连杆)的角度向量
        # 所以我们需要在首尾填充0
        fk_q1 = np.concatenate(([0], q1, [0]))
        fk_q2 = np.concatenate(([0], q2, [0]))

        # --- 正向运动学计算 ---
        # T_B1_E1: 左臂基座到末端的变换
        T_B1_E1 = left_chain.forward_kinematics(fk_q1)
        # T_B2_E2: 右臂基座到末端的变换
        T_B2_E2 = right_chain.forward_kinematics(fk_q2)

        # T_W_E1: 世界坐标系到左臂末端的变换
        T_W_E1 = T_W_B1 @ T_B1_E1
        # T_W_E2: 世界坐标系到右臂末端的变换
        T_W_E2 = T_W_B2 @ T_B2_E2

        # --- 计算误差 ---
        # 1. 任务空间误差: 虚拟物体中点M的当前位姿与目标位姿的差距
        # M的当前位姿可以由任一手臂计算得出，这里用左臂
        T_W_M_from_E1 = T_W_E1 @ T_E1_M
        
        # 位置误差: 计算两个点之间的欧氏距离
        pos_error = np.linalg.norm(T_W_M_from_E1[:3, 3] - target_pose_M[:3, 3])
        
        # 姿态误差: 计算两个姿态四元数之间的点积。点积绝对值越接近1，姿态越接近。
        orient_error_quat = Rotation.from_matrix(T_W_M_from_E1[:3,:3]).as_quat()
        target_orient_quat = Rotation.from_matrix(target_pose_M[:3,:3]).as_quat()
        orient_error = 1.0 - np.abs(np.dot(orient_error_quat, target_orient_quat))


        # 2. 闭环约束误差: 两个手臂末端当前的相对位姿与预设的恒定位姿的差距
        T_E1_E2_current = np.linalg.inv(T_W_E1) @ T_W_E2
        
        # 约束的位置误差
        constraint_pos_error = np.linalg.norm(T_E1_E2_current[:3, 3] - T_E1_E2[:3, 3])
        
        # 约束的姿态误差
        constraint_rot_current = Rotation.from_matrix(T_E1_E2_current[:3, :3]).as_quat()
        constraint_rot_target = Rotation.from_matrix(T_E1_E2[:3, :3]).as_quat()
        constraint_orient_error = 1.0 - np.abs(np.dot(constraint_rot_current, constraint_rot_target))

        # --- 组合总误差 ---
        # 最终的误差是任务空间误差和闭环约束误差的加权和。
        # 我们给闭环约束误差一个很大的权重(100)，因为这个约束必须被严格遵守。
        error = pos_error + orient_error + 100 * (constraint_pos_error + constraint_orient_error)
        return error

    # 从运动学链中为活动关节提取关节限制 (bounds)
    bounds_l = [l.bounds for l, active in zip(left_chain.links, left_chain.active_links_mask) if active]
    bounds_r = [l.bounds for l, active in zip(right_chain.links, right_chain.active_links_mask) if active]
    bounds = bounds_l + bounds_r

    # 使用Scipy的minimize函数进行优化
    result = minimize(
        objective,      # 目标函数 (我们要最小化的)
        initial_q,      # 初始猜测值
        method='SLSQP', # 一种支持约束和边界的优化算法
        bounds=bounds,  # 关节角度的边界
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-6} # 优化器选项
    )

    if result.success:
        print("IK solution found.")
        return result.x # 返回找到的关节角度
    else:
        print("IK solution not found.")
        print(result.message)
        return None

if __name__ == '__main__':
    xml_file = 'dual_elfin15_scene_clean.xml'
    
    # --- 1. 设置 ---
    print("Parsing XML and creating kinematic chains...")
    kinematics_data = get_kinematics(xml_file)
    left_chain, right_chain, T_W_B1, T_W_B2 = kinematics_data
    print("Chains created.")
    print(f"Left arm active links mask: {left_chain.active_links_mask}")
    print(f"Right arm active links mask: {right_chain.active_links_mask}")
    
    # 运动学的起点定义为两个基座连线的中点。
    # 在这个XML文件中，两个基座位于同一位置，所以中点就是那个位置。
    p_B1 = T_W_B1[:3, 3]
    p_B2 = T_W_B2[:3, 3]
    kinematic_origin = (p_B1 + p_B2) / 2
    print(f"Kinematic origin (midpoint of bases): {kinematic_origin}")


    # --- 2. 定义目标 ---
    # 定义虚拟物体中点M的目标位姿。
    # 比如，在运动学起点前方0.5m, 上方0.2m，并绕Z轴旋转45度。
    target_pose_M = np.eye(4)
    target_pose_M[:3, 3] = kinematic_origin + np.array([0.5, 0, 0.2])
    target_pose_M[:3, :3] = Rotation.from_euler('z', 45, degrees=True).as_matrix()

    print("\nTarget pose for object midpoint (T_W_M):")
    print(target_pose_M)

    # --- 3. 求解 ---
    # 从XML的'home'关键帧读取关节角度，将其作为优化的初始猜测值。
    # 一个好的初始值对优化至关重要。
    tree = ET.parse(xml_file)
    root = tree.getroot()
    home_qpos_str = root.find(".//key[@name='home']").get('qpos')
    initial_q = np.fromstring(home_qpos_str, sep=' ')
    
    print(f"\nUsing initial guess from 'home' keyframe: {initial_q}")

    print("\nSolving dual-arm IK...")
    solution_q = solve_dual_arm_ik(target_pose_M, initial_q, kinematics_data)

    # --- 4. 验证 ---
    if solution_q is not None:
        print("\nVerification of the solution:")
        q1_sol = solution_q[:6]
        q2_sol = solution_q[6:]
        print(f"Solution q1 (left arm): {np.rad2deg(q1_sol)}")
        print(f"Solution q2 (right arm): {np.rad2deg(q2_sol)}")

        # 使用找到的解进行正向运动学，计算实际到达的位姿
        fk_q1_sol = np.concatenate(([0], q1_sol, [0]))
        fk_q2_sol = np.concatenate(([0], q2_sol, [0]))
        
        T_B1_E1_sol = left_chain.forward_kinematics(fk_q1_sol)
        T_B2_E2_sol = right_chain.forward_kinematics(fk_q2_sol)

        T_W_E1_sol = T_W_B1 @ T_B1_E1_sol
        T_W_E2_sol = T_W_B2 @ T_B2_E2_sol

        T_E1_M = np.eye(4); T_E1_M[2, 3] = 3.3
        T_W_M_sol = T_W_E1_sol @ T_E1_M

        print("\nResulting Pose of object midpoint from solution:")
        print(T_W_M_sol)
        
        # 计算实际位姿与目标位姿的差距
        pos_diff = np.linalg.norm(T_W_M_sol[:3,3] - target_pose_M[:3,3])
        print(f"\nPosition difference to target: {pos_diff:.6f} meters")

        # 验证闭环约束是否仍然满足
        T_E2_M = np.eye(4); T_E2_M[2, 3] = 0.3
        # T_E2_M[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        T_E1_E2_target = T_E1_M @ np.linalg.inv(T_E2_M)
        T_E1_E2_sol = np.linalg.inv(T_W_E1_sol) @ T_W_E2_sol
        
        constraint_pos_diff = np.linalg.norm(T_E1_E2_sol[:3,3] - T_E1_E2_target[:3,3])
        print(f"Closed-loop constraint position difference: {constraint_pos_diff:.6f} meters")

        # --- 5. Visualization with MuJoCo ---
        print("\nAttempting to display the result directly in MuJoCo Viewer...")
        try:
            import mujoco
            import mujoco.viewer
            import time

            model = mujoco.MjModel.from_xml_path(xml_file)
            data = mujoco.MjData(model)

            # 将计算出的关节角度设置为模型的姿态
            data.qpos[:12] = solution_q

            # 使用官方推荐的 'with' 语句启动查看器
            # 这会自动处理查看器的生命周期
            print("Launching viewer...")
            with mujoco.viewer.launch_passive(model, data) as viewer:
                print("Viewer launched. It will close automatically when the script ends.")
                # 让查看器显示5秒钟
                time.sleep(5)


        except ImportError:
            print("\nCould not import 'mujoco'.")
            print("Please install it to visualize the result: pip install mujoco")
        except Exception as e:
            print(f"\nAn error occurred during visualization: {e}")
            print("Falling back to saving the result to an XML file.")
            # 后备方案：如果直接可视化失败，则保存到XML
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                home_key = root.find(".//key[@name='home']")
                if home_key is not None:
                    solution_q_str = " ".join(map(str, solution_q))
                    home_key.set('qpos', solution_q_str)
                    home_key.set('ctrl', solution_q_str)
                
                result_xml_file = 'dual_elfin15_scene_result.xml'
                tree.write(result_xml_file, encoding='utf-8', xml_declaration=True)
                
                print(f"\nResult saved to '{result_xml_file}'.")
                print(f"You can view it by running: simulate {result_xml_file}")
            except Exception as xml_e:
                print(f"An error occurred during XML fallback as well: {xml_e}") 