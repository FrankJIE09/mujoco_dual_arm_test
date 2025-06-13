import numpy as np


class JointTransform:
    """
    机器人关节角度转换工具类
    用于主从臂之间的关节角度转换
    """

    @staticmethod
    def normalize_angle(angle_deg):
        """
        将角度规范化到 [-360, 360] 度范围内
        
        Args:
            angle_deg (float): 输入角度（度数）
            
        Returns:
            float: 规范化后的角度（度数）
        """
        while angle_deg > 360.0:
            angle_deg -= 360.0
        while angle_deg < -360.0:
            angle_deg += 360.0
        return angle_deg

    @staticmethod
    def normalize_angle_radians(angle_rad):
        """
        将弧度角度规范化到 [-π, π] 范围内
        
        Args:
            angle_rad (float): 输入角度（弧度）
            
        Returns:
            float: 规范化后的角度（弧度）
        """
        while angle_rad > np.pi:
            angle_rad -= 2 * np.pi
        while angle_rad <= -np.pi:
            angle_rad += 2 * np.pi
        return angle_rad

    @staticmethod
    def transform_leader_to_follower_joints(joint1_rad, joint2_rad, joint3_rad, 
                                          joint4_rad, joint5_rad, joint6_rad):
        """
        将主臂关节角度转换为从臂关节角度
        
        Args:
            joint1_rad, joint2_rad, joint3_rad, joint4_rad, joint5_rad, joint6_rad (float): 
                主臂的6个关节角度（弧度）
            
        Returns:
            tuple: 从臂的6个关节角度（弧度）
        """
        # 输入弧度转换为度数
        leader_joints = [
            np.degrees(joint1_rad),
            np.degrees(joint2_rad), 
            np.degrees(joint3_rad),
            np.degrees(joint4_rad),
            np.degrees(joint5_rad),
            np.degrees(joint6_rad)
        ]
        
        # 初始化从臂关节角度数组
        follower_joints = [0.0] * 6
        
        # 更新后的转换规则 (度数计算):
        # Joint 1 (index 0): -angle
        follower_joints[0] = JointTransform.normalize_angle(-leader_joints[0])
        # Joint 2 (index 1): -180-angle  
        follower_joints[1] = JointTransform.normalize_angle(-180.0 - leader_joints[1] + 180)
        # Joint 3 (index 2): -angle
        follower_joints[2] = JointTransform.normalize_angle(-leader_joints[2])
        # Joint 4 (index 3): -180 - angle
        follower_joints[3] = JointTransform.normalize_angle(-180.0 - leader_joints[3])
        # Joint 5 (index 4): -360 - angle
        follower_joints[4] = JointTransform.normalize_angle(-360.0 + leader_joints[4])
        # Joint 6 (index 5): -180 - angle
        follower_joints[5] = JointTransform.normalize_angle(-180.0 - leader_joints[5])
        
        # 转换回弧度并规范化到 [-π, π] 范围内
        follower_joints_rad = [np.radians(angle) for angle in follower_joints]
        normalized_joints_rad = [JointTransform.normalize_angle_radians(angle) for angle in follower_joints_rad]
        return tuple(normalized_joints_rad)

    @staticmethod
    def transform_joints_array(leader_joints_rad):
        """
        将主臂关节角度数组转换为从臂关节角度数组
        
        Args:
            leader_joints_rad (list or np.ndarray): 主臂的6个关节角度（弧度）
            
        Returns:
            np.ndarray: 从臂的6个关节角度（弧度）
        """
        if len(leader_joints_rad) != 6:
            raise ValueError("输入的关节角度数组长度必须为6")
        
        follower_joints = JointTransform.transform_leader_to_follower_joints(*leader_joints_rad)
        return np.array(follower_joints)


def test_joint_transform():
    """
    测试主从臂关节角度转换功能
    """
    print("--- 主从臂关节角度转换测试 ---")
    
    # 示例主臂关节角度 (弧度)
    leader_angles_rad = [0.14403416, -0.62792113, 1.70052709, 3.14, -0.81314359, 0.14512927]
    print(f"主臂关节角度 (弧度): {leader_angles_rad}")
    print(f"主臂关节角度 (度): {np.degrees(leader_angles_rad).tolist()}")
    
    # 使用转换函数
    follower_angles_rad = JointTransform.transform_leader_to_follower_joints(*leader_angles_rad)
    print(f"从臂关节角度 (弧度): {list(follower_angles_rad)}")
    print(f"从臂关节角度 (度): {np.degrees(follower_angles_rad).tolist()}")
    
    # 测试数组形式的转换
    print("\n--- 使用数组形式转换 ---")
    follower_array = JointTransform.transform_joints_array(leader_angles_rad)
    print(f"从臂关节角度数组 (弧度): {follower_array}")
    print(f"从臂关节角度数组 (度): {np.degrees(follower_array).tolist()}")


if __name__ == '__main__':
    test_joint_transform() 