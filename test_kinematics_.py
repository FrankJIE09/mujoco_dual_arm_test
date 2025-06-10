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
    print("--- 正向运动学示例 ---")
    # 注意：ikpy 和 MuJoCo 对关节的定义和数量可能不同。
    # ikpy 的 solution 会包含非活动关节的0。
    # 我们用一个已知的、合理的关节角度来做FK
    joint_angles_fk = np.array([0., 0, 0, 0, 0.1, 0])
    
    # FK在MuJoCo中计算
    position_fk, orientation_fk_quat = kin.forward_kinematics(joint_angles_fk)
    print(f"  给定关节角: {joint_angles_fk}")
    print(f"  计算出的末端位置: {position_fk}")
    print(f"  计算出的末端姿态 (四元数): {orientation_fk_quat}")
    print("-" * 30)


    # --- 逆向运动学 (IK) 示例 ---
    print("--- 逆向运动学示例 ---")
    # 将FK计算出的位置和姿态作为IK的目标
    target_position_ik = position_fk.copy()

    # ikpy 需要旋转矩阵作为姿态目标，我们从 MuJoCo 的四元数转换得到
    from scipy.spatial.transform import Rotation
    target_orientation_ik_quat = orientation_fk_quat.copy()
    # MuJoCo 的四元数是 [w, x, y, z], SciPy 是 [x, y, z, w]
    scipy_quat = np.roll(target_orientation_ik_quat, -1)
    target_orientation_ik_matrix = Rotation.from_quat(scipy_quat).as_matrix()

    print(f"  目标位置: {target_position_ik}")
    print(f"  目标姿态 (旋转矩阵):\n{target_orientation_ik_matrix}")

    # 计算逆向运动学解
    # ikpy的解包括所有连杆，我们需要提取出活动关节
    ik_solution_full = kin.inverse_kinematics(
        target_pos=target_position_ik,
        target_orientation_matrix=target_orientation_ik_matrix
    )

    
    # 由于我们设置了正确的活动关节掩码，现在固定关节的值应该为0
    # 提取索引2-7的关节角度（对应6个旋转驱动关节）
    ik_solution_mujoco = ik_solution_full[2:8]  # 提取索引2-7的关节角度

    print(f"\nikpy 完整解 (9个关节): {ik_solution_full}")
    print(f"MuJoCo 需要的解 (6个驱动关节): {ik_solution_mujoco}")
    print(f"固定关节值检查 - 索引0: {ik_solution_full[0]:.6f}, 索引1: {ik_solution_full[1]:.6f}, 索引8: {ik_solution_full[8]:.6f}")

    # --- 验证与可视化 ---
    # 使用IK解算出的关节角度进行FK，验证其准确性
    # 注意：ikpy的解可能与原始关节角不完全一样，但对应的末端位置应非常接近
    final_pos_fk, _ = kin.forward_kinematics(ik_solution_mujoco)
    print("\n验证结果:")
    print(f"  FK计算出的原始位置: {position_fk}")
    print(f"  IK解进行FK后的位置: {final_pos_fk}")
    print(f"  位置误差: {np.linalg.norm(position_fk - final_pos_fk)}")

    # 可视化IK解对应的机器人姿态
    kin.visualize(ik_solution_mujoco)