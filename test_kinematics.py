import numpy as np
import mujoco
import mujoco.viewer
import time
import os
from ikpy.chain import Chain

class Elfin15Kinematics:
    """
    该类封装了 Elfin15 机器人的运动学计算功能。
    - 正向运动学 (FK) 和可视化使用 MuJoCo。
    - 逆向运动学 (IK) 使用 ikpy 库，从 URDF 文件加载。
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
        


    def forward_kinematics(self, joint_angles):
        """
        计算机器人的正向运动学（Forward Kinematics, FK）。
        使用 MuJoCo 进行计算。

        Args:
            joint_angles (list or np.ndarray): 包含机器人所有关节角度的列表或数组。
                                               注意：ikpy的关节从0开始，而MuJoCo可能包含非驱动关节。
                                               这里我们假设输入的 joint_angles 是驱动关节的角度。

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
        使用 ikpy 库进行计算。

        Args:
            target_pos (list or np.ndarray): 末端执行器的目标位置 [x, y, z]。
            target_orientation_matrix (np.ndarray, optional): 3x3 的目标姿态旋转矩阵。
            initial_position (list, optional): 求解器开始迭代的初始关节位置。
                                               长度应与活动关节数相同。默认为零位姿。

        Returns:
            np.ndarray: 计算出的关节角度数组。ikpy 返回的数组包含所有关节（包括固定关节），
                        我们需要提取活动关节。
        """
        # 如果没有提供初始位置，使用零向量作为初始猜测
        if initial_position is None:
            # ikpy 链的总长度
            initial_position = [0.0] * len(self.ik_chain.links)
        
        # 使用 ikpy 计算IK
        # ikpy 的 'all' 模式会同时考虑位置和姿态
        ik_solution = self.ik_chain.inverse_kinematics(
            target_position=target_pos,
            target_orientation=target_orientation_matrix,
            orientation_mode="all",
            initial_position=initial_position
        )
        return ik_solution

if __name__ == '__main__':
    # --- 初始化 ---
    kin = Elfin15Kinematics()

    # --- 正向运动学 (FK) 示例 ---
    print("--- 正向运动学示例 (使用ikpy) ---")
    # 定义测试关节角度 (ikpy格式，9个关节)
    joint_angles_fk_ikpy = np.array([0., 0., 0, 0, 0, -0, 0.4, 0., 0.])
    
    # FK在ikpy中计算
    transform_matrix_fk = kin.ik_chain.forward_kinematics(joint_angles_fk_ikpy)
    position_fk = transform_matrix_fk[:3, 3]
    orientation_fk_matrix = transform_matrix_fk[:3, :3]
    
    print(f"  给定关节角 (ikpy格式): {joint_angles_fk_ikpy}")
    print(f"  计算出的末端位置: {position_fk}")
    print(f"  计算出的末端姿态 (旋转矩阵):\n{orientation_fk_matrix}")
    print("-" * 30)


    # --- 逆向运动学 (IK) 示例 ---
    print("--- 逆向运动学示例 (使用ikpy) ---")
    # 将ikpy FK计算出的位置和姿态作为IK的目标
    target_position_ik = position_fk.copy()
    target_orientation_ik_matrix = orientation_fk_matrix.copy()

    print(f"  目标位置: {target_position_ik}")
    print(f"  目标姿态 (旋转矩阵):\n{target_orientation_ik_matrix}")

    # 计算逆向运动学解 (直接使用ikpy，无坐标转换)
    # 使用ikpy库的原始接口
    ik_solution_full = kin.ik_chain.inverse_kinematics(
        target_position=target_position_ik,
        target_orientation=target_orientation_ik_matrix,
        orientation_mode="all"
    )

    
    # 由于我们设置了正确的活动关节掩码，现在固定关节的值应该为0
    # 提取索引2-7的关节角度（对应6个旋转驱动关节）
    ik_solution_mujoco = ik_solution_full[2:8]  # 提取索引2-7的关节角度

    print(f"\nikpy 完整解 (9个关节): {ik_solution_full}")
    print(f"MuJoCo 需要的解 (6个驱动关节): {ik_solution_mujoco}")
    print(f"固定关节值检查 - 索引0: {ik_solution_full[0]:.6f}, 索引1: {ik_solution_full[1]:.6f}, 索引8: {ik_solution_full[8]:.6f}")

    # --- ikpy内部一致性验证 ---
    # 使用IK解算出的关节角度在ikpy中进行FK验证
    transform_matrix_verify = kin.ik_chain.forward_kinematics(ik_solution_full)
    final_pos_ikpy = transform_matrix_verify[:3, 3]
    final_orientation_ikpy = transform_matrix_verify[:3, :3]
    
    print("\nikpy内部一致性验证:")
    print(f"  原始FK位置: {position_fk}")
    print(f"  IK解FK位置: {final_pos_ikpy}")
    print(f"  位置误差: {np.linalg.norm(position_fk - final_pos_ikpy):.10f}")
    
    print(f"  原始关节角度: {joint_angles_fk_ikpy}")
    print(f"  IK求解角度: {ik_solution_full}")
    print(f"  关节角度差异: {np.linalg.norm(joint_angles_fk_ikpy - ik_solution_full):.10f}")
    
    # 检查IK是否完美恢复了原始配置
    if np.linalg.norm(position_fk - final_pos_ikpy) < 1e-10:
        print("  ✅ ikpy FK/IK完美一致!")
    else:
        print("  ⚠️  ikpy FK/IK存在小误差")

    # --- MuJoCo可视化展示 ---
    print("\n--- MuJoCo可视化展示 ---")
    # 提取MuJoCo需要的6个驱动关节角度进行展示
    ik_solution_mujoco = ik_solution_full[2:8]  # 提取索引2-7的关节角度
    print(f"MuJoCo展示用关节角度 (6个): {ik_solution_mujoco}")
    
    # 可视化IK解对应的机器人姿态
    print("启动MuJoCo可视化...")
    kin.visualize(ik_solution_mujoco)