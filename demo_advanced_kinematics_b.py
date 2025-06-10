#!/usr/bin/env python3
"""
Elfin15 高级运动学功能演示

该脚本演示了扩展的运动学功能：
1. 雅可比矩阵计算和奇异性检测
2. 工作空间分析和可视化
3. 直线轨迹规划
4. 轨迹动画执行
5. 关节限制检查
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kinematics import Elfin15Kinematics


def demo_jacobian_analysis(kin):
    """演示雅可比矩阵分析和奇异性检测"""
    print("=" * 50)
    print("🧮 雅可比矩阵分析和奇异性检测")
    print("=" * 50)

    # 测试不同的关节配置
    test_configs = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 零位姿
        [0.5, 0.2, -0.3, 0.1, 0.4, 0.2],  # 随机配置1
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 随机配置2
        [0.0, 1.57, 0.0, 0.0, 0.0, 0.0],  # 可能的奇异配置
    ]

    config_names = ["零位姿", "随机配置1", "随机配置2", "奇异配置测试"]

    for i, (config, name) in enumerate(zip(test_configs, config_names)):
        print(f"\n{i + 1}. {name}: {config}")

        try:
            # 计算雅可比矩阵
            jacp, jacr = kin.compute_jacobian(config)

            # 计算奇异值
            _, s_pos, _ = np.linalg.svd(jacp)
            min_singular_value = np.min(s_pos)

            # 检查奇异性
            is_singular = kin.check_singularity(config)

            print(f"   位置雅可比矩阵形状: {jacp.shape}")
            print(f"   最小奇异值: {min_singular_value:.6f}")
            print(f"   是否接近奇异性: {is_singular}")

            # 计算末端位置
            pos, quat = kin.forward_kinematics(config)
            print(f"   末端位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        except Exception as e:
            print(f"   ❌ 计算失败: {e}")


def demo_workspace_analysis(kin):
    """演示工作空间分析"""
    print("\n" + "=" * 50)
    print("🌐 工作空间分析")
    print("=" * 50)

    # 生成工作空间点云
    print("正在生成工作空间点云...")
    workspace_points = kin.generate_workspace_points(num_samples=2000)

    # 计算工作空间统计信息
    x_range = [np.min(workspace_points[:, 0]), np.max(workspace_points[:, 0])]
    y_range = [np.min(workspace_points[:, 1]), np.max(workspace_points[:, 1])]
    z_range = [np.min(workspace_points[:, 2]), np.max(workspace_points[:, 2])]

    print(f"工作空间范围:")
    print(f"  X: [{x_range[0]:.3f}, {x_range[1]:.3f}] m")
    print(f"  Y: [{y_range[0]:.3f}, {y_range[1]:.3f}] m")
    print(f"  Z: [{z_range[0]:.3f}, {z_range[1]:.3f}] m")

    # 计算工作空间体积（近似）
    volume = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0]) * (z_range[1] - z_range[0])
    print(f"  近似工作空间体积: {volume:.3f} m³")

    # 尝试绘制工作空间
    try:
        print("\n正在绘制工作空间...")
        kin.plot_workspace(num_samples=2000)
        print("✅ 工作空间图已保存!")
    except Exception as e:
        print(f"⚠️  工作空间绘制失败: {e}")


def demo_trajectory_planning(kin):
    """演示轨迹规划功能"""
    print("\n" + "=" * 50)
    print("🛤️  轨迹规划演示")
    print("=" * 50)

    # 定义几个测试轨迹
    trajectories = [
        {
            "name": "短距离轨迹",
            "start": np.array([0.4, 0.2, 1.3]),
            "end": np.array([0.4, -0.2, 1.3]),
            "points": 15
        },
        {
            "name": "垂直轨迹",
            "start": np.array([0.3, 0.0, 1.2]),
            "end": np.array([0.3, 0.0, 1.5]),
            "points": 20
        },
        {
            "name": "对角线轨迹",
            "start": np.array([0.2, 0.3, 1.1]),
            "end": np.array([0.5, -0.1, 1.4]),
            "points": 25
        }
    ]

    successful_trajectories = []

    for i, traj in enumerate(trajectories):
        print(f"\n{i + 1}. {traj['name']}")
        print(f"   起始: {traj['start']}")
        print(f"   结束: {traj['end']}")
        print(f"   轨迹点数: {traj['points']}")

        # 规划轨迹
        trajectory_points, joint_trajectory, success = kin.plan_straight_line_trajectory(
            traj['start'], traj['end'], num_points=traj['points']
        )

        if success:
            print(f"   ✅ 轨迹规划成功!")
            print(f"   关节轨迹形状: {joint_trajectory.shape}")

            # 计算轨迹统计信息
            joint_changes = np.diff(joint_trajectory, axis=0)
            max_joint_change = np.max(np.abs(joint_changes))
            avg_joint_change = np.mean(np.abs(joint_changes))

            print(f"   最大关节变化: {max_joint_change:.4f} rad")
            print(f"   平均关节变化: {avg_joint_change:.4f} rad")

            successful_trajectories.append({
                'name': traj['name'],
                'joint_trajectory': joint_trajectory,
                'trajectory_points': trajectory_points
            })
        else:
            print(f"   ❌ 轨迹规划失败!")

    return successful_trajectories


def demo_joint_limits(kin):
    """演示关节限制功能"""
    print("\n" + "=" * 50)
    print("⚙️  关节限制分析")
    print("=" * 50)

    print("关节限制信息:")
    for i, limits in enumerate(kin.joint_limits):
        range_deg = np.degrees([limits[0], limits[1]])
        print(f"  关节 {i + 1}: [{limits[0]:.3f}, {limits[1]:.3f}] rad "
              f"= [{range_deg[0]:.1f}°, {range_deg[1]:.1f}°]")

    # 测试随机配置的关节限制检查
    print(f"\n测试随机关节配置:")
    for i in range(5):
        # 生成随机关节角度（可能超出限制）
        random_angles = np.random.uniform(-np.pi, np.pi, 6)
        within_limits = kin._check_joint_limits(random_angles)

        print(f"  配置 {i + 1}: {within_limits} - {random_angles}")


def interactive_demo():
    """交互式演示菜单"""
    print("🤖 Elfin15 高级运动学功能演示")
    print("=" * 50)

    # 初始化运动学对象
    try:
        kin = Elfin15Kinematics()
        print("✅ 运动学系统初始化成功!")
    except Exception as e:
        print(f"❌ 运动学系统初始化失败: {e}")
        return

    while True:
        print("\n" + "=" * 30)
        print("请选择演示功能:")
        print("1. 雅可比矩阵分析和奇异性检测")
        print("2. 工作空间分析")
        print("3. 轨迹规划演示")
        print("4. 关节限制分析")
        print("5. 执行所有演示")
        print("0. 退出")
        print("=" * 30)

        try:
            choice = input("请输入选择 (0-5): ").strip()

            if choice == '0':
                print("👋 再见!")
                break
            elif choice == '1':
                demo_jacobian_analysis(kin)
            elif choice == '2':
                demo_workspace_analysis(kin)
            elif choice == '3':
                successful_trajs = demo_trajectory_planning(kin)
                # 询问是否执行动画
                if successful_trajs:
                    show_anim = input("\n是否显示轨迹动画? (y/n): ").lower().strip()
                    if show_anim == 'y':
                        for traj in successful_trajs:
                            print(f"播放轨迹: {traj['name']}")
                            kin.animate_trajectory(traj['joint_trajectory'], dt=0.1)
            elif choice == '4':
                demo_joint_limits(kin)
            elif choice == '5':
                demo_jacobian_analysis(kin)
                demo_workspace_analysis(kin)
                successful_trajs = demo_trajectory_planning(kin)
                demo_joint_limits(kin)
                print("\n🎉 所有演示完成!")
            else:
                print("❌ 无效选择，请重新输入!")

        except KeyboardInterrupt:
            print("\n👋 用户中断，退出程序!")
            break
        except Exception as e:
            print(f"❌ 演示执行出错: {e}")


if __name__ == '__main__':
    interactive_demo()