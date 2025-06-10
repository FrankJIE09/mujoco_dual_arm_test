import numpy as np
import mujoco
import mujoco.viewer
import time
import os
from ikpy.chain import Chain
import matplotlib.pyplot as plt


class Elfin15Kinematics:
    """
    该类封装了 Elfin15 机器人的运动学计算功能。
    - 正向运动学 (FK) 和可视化使用 MuJoCo。
    - 逆向运动学 (IK) 使用 ikpy 库，从 URDF 文件加载。
    - 新增功能：雅可比矩阵、工作空间分析、轨迹规划
    """

    def __init__(self, mjcf_path=None, urdf_path=None):
        """
        初始化 Elfin15Kinematics 类的实例。

        Args:
            mjcf_path (str, optional): MJCF 模型文件的路径。如果为 None，则自动构造。
            urdf_path (str, optional): URDF 模型文件的路径。如果为 None，则自动构造。
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # --- 配置 MuJoCo (用于FK和可视化) ---
        if mjcf_path is None:
            mjcf_path = os.path.join(script_dir, "mjcf_models", "elfin15", "elfin15.xml")
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.end_effector_name = "elfin_link6"

        # --- 配置 ikpy (用于IK) ---
        if urdf_path is None:
            urdf_path = os.path.join(script_dir, "mjcf_models", "elfin15", "elfin15.urdf")
        
        # 指定正确的基础连杆名称和活动关节掩码
        # 只让索引2-7的6个旋转关节为活动，其他固定关节设为False
        active_mask = [False, False, True, True, True, True, True, True, False]
        self.ik_chain = Chain.from_urdf_file(
            urdf_path, 
            base_elements=["elfin_base_link"],
            active_links_mask=active_mask
        )

        # 获取关节限制
        self.joint_limits = self._get_joint_limits()

        # 计算坐标系转换参数
        self._calibrate_coordinate_transform()

        # 调试信息：打印ikpy链的结构
        print(f"🔍 ikpy链结构调试信息：")
        print(f"  总链长度: {len(self.ik_chain.links)}")
        print(f"  活动链接掩码: {self.ik_chain.active_links_mask}")
        print(f"  链接详情:")
        for i, link in enumerate(self.ik_chain.links):
            try:
                joint_type = getattr(link, 'joint_type', 'unknown')
                link_name = getattr(link, 'name', f'Link_{i}')
                is_active = self.ik_chain.active_links_mask[i]
                print(f"    {i}: {link_name} (关节类型: {joint_type}, 活动: {is_active})")
            except Exception as e:
                print(f"    {i}: 链接信息获取失败 - {e}")
        print("-" * 40)

    def _get_joint_limits(self):
        """获取关节限制"""
        joint_limits = []
        for i in range(self.model.nq):
            qpos_min = self.model.jnt_range[i, 0] if self.model.jnt_limited[i] else -np.pi
            qpos_max = self.model.jnt_range[i, 1] if self.model.jnt_limited[i] else np.pi
            joint_limits.append([qpos_min, qpos_max])
        return np.array(joint_limits)

    def _calibrate_coordinate_transform(self):
        """
        校准ikpy和MuJoCo之间的坐标系转换
        通过比较零位姿下的末端位置计算偏移
        """
        # 零位姿关节角度
        zero_joints = np.zeros(6)
        zero_joints_ikpy = np.zeros(9)  # ikpy需要9个关节，前后为固定关节
        zero_joints_ikpy[2:8] = zero_joints

        # 计算MuJoCo在零位姿下的末端位置
        mujoco_pos_zero, _ = self.forward_kinematics(zero_joints)

        # 计算ikpy在零位姿下的末端位置
        ikpy_transform = self.ik_chain.forward_kinematics(zero_joints_ikpy)
        ikpy_pos_zero = ikpy_transform[:3, 3]

        # 计算偏移量
        self.position_offset = mujoco_pos_zero - ikpy_pos_zero
        
        print(f"🔧 坐标系校准信息:")
        print(f"  MuJoCo零位末端位置: {mujoco_pos_zero}")
        print(f"  ikpy零位末端位置: {ikpy_pos_zero}")
        print(f"  计算的位置偏移: {self.position_offset}")

    def ikpy_forward_kinematics(self, joint_angles_full):
        """
        使用ikpy计算正向运动学，并应用坐标转换
        
        Args:
            joint_angles_full: ikpy格式的完整关节角度数组（9个元素）
            
        Returns:
            tuple: (修正后的位置, 旋转矩阵)
        """
        transform_matrix = self.ik_chain.forward_kinematics(joint_angles_full)
        position = transform_matrix[:3, 3] + self.position_offset  # 应用位置偏移
        rotation_matrix = transform_matrix[:3, :3]
        return position, rotation_matrix

    def forward_kinematics(self, joint_angles):
        """
        计算机器人的正向运动学（Forward Kinematics, FK）。
        使用 MuJoCo 进行计算。

        Args:
            joint_angles (list or np.ndarray): 包含机器人所有关节角度的列表或数组。

        Returns:
            tuple: 一个元组，包含：
                   - end_effector_pos (np.ndarray): 末端执行器的位置 (x, y, z)。
                   - end_effector_quat (np.ndarray): 末端执行器的姿态（四元数 w, x, y, z）。
        """
        if len(joint_angles) != self.model.nq:
            raise ValueError(f"期望的关节角度数量为 {self.model.nq}，但收到了 {len(joint_angles)}")

        self.data.qpos[:] = joint_angles
        mujoco.mj_forward(self.model, self.data)

        end_effector_pos = self.data.body(self.end_effector_name).xpos
        end_effector_quat = self.data.body(self.end_effector_name).xquat

        return end_effector_pos, end_effector_quat

    def compute_jacobian(self, joint_angles):
        """
        计算末端执行器的雅可比矩阵

        Args:
            joint_angles: 当前关节角度

        Returns:
            tuple: (位置雅可比矩阵 3xN, 姿态雅可比矩阵 3xN)
        """
        self.data.qpos[:] = joint_angles
        mujoco.mj_forward(self.model, self.data)

        # 获取末端执行器的body ID
        end_effector_id = self.model.body(self.end_effector_name).id

        # 计算雅可比矩阵
        jacp = np.zeros((3, self.model.nv))  # 位置雅可比
        jacr = np.zeros((3, self.model.nv))  # 姿态雅可比

        mujoco.mj_jac(self.model, self.data, jacp, jacr,
                      self.data.body(self.end_effector_name).xpos, end_effector_id)

        return jacp, jacr

    def check_singularity(self, joint_angles, threshold=1e-3):
        """
        检查当前配置是否接近奇异性

        Args:
            joint_angles: 关节角度
            threshold: 奇异性阈值

        Returns:
            bool: True表示接近奇异性
        """
        jacp, _ = self.compute_jacobian(joint_angles)
        # 计算雅可比矩阵的最小奇异值
        _, s, _ = np.linalg.svd(jacp)
        min_singular_value = np.min(s)
        return min_singular_value < threshold

    def inverse_kinematics_with_limits(self, target_pos, target_orientation_matrix=None,
                                       initial_position=None, max_iterations=1000):
        """
        带关节限制的逆向运动学求解

        Args:
            target_pos: 目标位置
            target_orientation_matrix: 目标姿态矩阵
            initial_position: 初始关节位置
            max_iterations: 最大迭代次数

        Returns:
            tuple: (关节角度解, 是否成功)
        """
        if initial_position is None:
            initial_position = [0.0] * len(self.ik_chain.links)

        try:
            # 将MuJoCo坐标系的目标位置转换为ikpy坐标系
            target_pos_ikpy = np.array(target_pos) - self.position_offset
            
            ik_solution = self.ik_chain.inverse_kinematics(
                target_position=target_pos_ikpy,
                target_orientation=target_orientation_matrix,
                orientation_mode="all",
                initial_position=initial_position,
                max_iter=max_iterations
            )

            # 提取驱动关节角度
            joint_angles = ik_solution[2:8]

            # 检查关节限制
            within_limits = self._check_joint_limits(joint_angles)

            return joint_angles, within_limits

        except Exception as e:
            print(f"IK求解失败: {e}")
            return None, False

    def _check_joint_limits(self, joint_angles):
        """检查关节角度是否在限制范围内"""
        for i, angle in enumerate(joint_angles):
            if angle < self.joint_limits[i, 0] or angle > self.joint_limits[i, 1]:
                return False
        return True

    def generate_workspace_points(self, num_samples=1000):
        """
        生成工作空间点云

        Args:
            num_samples: 采样点数

        Returns:
            np.ndarray: 工作空间点云
        """
        workspace_points = []

        for _ in range(num_samples):
            # 在关节限制范围内随机生成关节角度
            random_angles = []
            for i in range(6):
                angle = np.random.uniform(self.joint_limits[i, 0], self.joint_limits[i, 1])
                random_angles.append(angle)

            # 计算对应的末端位置
            pos, _ = self.forward_kinematics(random_angles)
            workspace_points.append(pos.copy())

        return np.array(workspace_points)

    def plot_workspace(self, num_samples=2000):
        """
        绘制机器人工作空间的3D图

        Args:
            num_samples: 采样点数
        """
        print(f"🔍 生成工作空间点云 ({num_samples} 个采样点)...")
        workspace_points = self.generate_workspace_points(num_samples)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制点云
        ax.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2],
                   c=workspace_points[:, 2], cmap='viridis', alpha=0.6, s=1)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Elfin15 机器人工作空间')

        # 添加颜色条
        plt.colorbar(ax.collections[0], ax=ax, shrink=0.8, label='Z 高度 (m)')

        plt.tight_layout()
        plt.savefig('elfin15_workspace.png', dpi=300, bbox_inches='tight')
        print("📊 工作空间图已保存为 'elfin15_workspace.png'")
        plt.show()

    def plan_straight_line_trajectory(self, start_pos, end_pos, num_points=50):
        """
        规划直线轨迹

        Args:
            start_pos: 起始位置
            end_pos: 结束位置
            num_points: 轨迹点数

        Returns:
            tuple: (轨迹点, 关节角度轨迹, 成功标志)
        """
        # 生成直线轨迹点
        trajectory_points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            point = start_pos + t * (end_pos - start_pos)
            trajectory_points.append(point)

        # 为每个轨迹点求解IK
        joint_trajectory = []
        initial_guess = [0.0] * len(self.ik_chain.links)

        success = True
        for i, point in enumerate(trajectory_points):
            joint_angles, ik_success = self.inverse_kinematics_with_limits(
                point, initial_position=initial_guess
            )

            if ik_success:
                joint_trajectory.append(joint_angles)
                # 使用当前解作为下一次的初始猜测，保证连续性
                initial_guess[2:8] = joint_angles
            else:
                print(f"⚠️  轨迹点 {i} IK求解失败")
                success = False
                break

        return np.array(trajectory_points), np.array(joint_trajectory), success

    def animate_trajectory(self, joint_trajectory, dt=0.1):
        """
        动画显示轨迹执行

        Args:
            joint_trajectory: 关节角度轨迹
            dt: 时间步长
        """
        print("🎬 开始轨迹动画...")

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for i, joint_angles in enumerate(joint_trajectory):
                self.data.qpos[:] = joint_angles
                mujoco.mj_forward(self.model, self.data)
                viewer.sync()
                time.sleep(dt)

                if not viewer.is_running():
                    break

                if i % 10 == 0:
                    print(f"执行进度: {i + 1}/{len(joint_trajectory)}")

        print("轨迹动画完成")

    def visualize(self, joint_angles):
        """
        根据给定的关节角度，使用 MuJoCo 可视化机器人的姿态。
        """
        if len(joint_angles) != self.model.nq:
            raise ValueError(f"期望的关节角度数量为 {self.model.nq}，但收到了 {len(joint_angles)}")

        self.data.qpos[:] = joint_angles
        mujoco.mj_forward(self.model, self.data)

        print("🚀 启动可视化窗口...")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            while viewer.is_running() and time.time() - start_time < 10:
                viewer.sync()
            viewer.close()
        print("可视化窗口已关闭。")

    def inverse_kinematics(self, target_pos, target_orientation_matrix=None, initial_position=None):
        """
        计算机器人的逆向运动学（Inverse Kinematics, IK）。
        使用 ikpy 库进行计算，并应用坐标系转换。

        Args:
            target_pos (list or np.ndarray): 末端执行器的目标位置 [x, y, z]（MuJoCo坐标系）。
            target_orientation_matrix (np.ndarray, optional): 3x3 的目标姿态旋转矩阵。
            initial_position (list, optional): 求解器开始迭代的初始关节位置。

        Returns:
            np.ndarray: 计算出的关节角度数组（完整9个关节）。
        """
        if initial_position is None:
            initial_position = [0.0] * len(self.ik_chain.links)

        # 将MuJoCo坐标系的目标位置转换为ikpy坐标系
        target_pos_ikpy = np.array(target_pos) - self.position_offset

        ik_solution = self.ik_chain.inverse_kinematics(
            target_position=target_pos_ikpy,
            target_orientation=target_orientation_matrix,
            orientation_mode="all",
            initial_position=initial_position
        )
        return ik_solution

