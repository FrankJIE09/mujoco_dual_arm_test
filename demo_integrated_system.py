#!/usr/bin/env python3
"""
双臂机器人完整系统演示

该脚本演示了从路径规划到仿真的完整流程：
1. 贝塞尔曲线路径规划
2. 球面线性插值姿态插值
3. 逆运动学求解
4. 轨迹平滑
5. MuJoCo仿真

作者: frank
日期: 2024年6月19日
"""

import numpy as np
from path_planner import BezierPathPlanner
from mujoco_simulator import DualArmSimulator


def main():
    """
    主函数：演示完整的路径规划到仿真系统
    """
    print("=" * 70)
    print("🤖 双臂机器人完整系统演示")
    print("   贝塞尔曲线路径规划 + 球面插值 + MuJoCo仿真")
    print("=" * 70)
    
    # 配置参数
    xml_file = 'dual_elfin15_scene_clean.xml'
    
    # 第一部分：路径规划和轨迹生成
    print("\n📈 第一部分：路径规划和轨迹生成")
    print("-" * 50)
    
    try:
        # 1. 初始化路径规划器
        planner = BezierPathPlanner(xml_file)
        
        # 2. 定义任务
        print(f"\n🎯 定义路径规划任务")
        start_pose = planner.T_W_M_home.copy()
        end_pose = start_pose.copy()
        end_pose[:3, 3] += np.array([0.3, 0.1, 0.2])  # 向右30cm，向前10cm，向上20cm
        
        print(f"   起始位置: {start_pose[:3, 3]}")
        print(f"   目标位置: {end_pose[:3, 3]}")
        print(f"   移动距离: {np.linalg.norm(end_pose[:3, 3] - start_pose[:3, 3]):.3f}m")
        
        # 3. 规划路径
        print(f"\n🚀 生成贝塞尔曲线路径")
        path_poses = planner.plan_bezier_path(start_pose, end_pose, num_points=30)
        
        # 4. 可视化路径
        print(f"\n📊 3D路径可视化")
        planner.visualize_path_3d(path_poses, save_path='integrated_path_3d.png')
        
        # 5. 求解逆运动学
        print(f"\n🔧 求解逆运动学")
        joint_trajectory, successful_solves = planner.solve_ik_for_path(path_poses)
        
        if successful_solves == 0:
            print("❌ 没有成功求解任何路径点，退出演示")
            return
            
        success_rate = (successful_solves / len(path_poses)) * 100
        print(f"   成功率: {success_rate:.1f}% ({successful_solves}/{len(path_poses)})")
        
        # 6. 平滑轨迹
        print(f"\n🌊 轨迹平滑处理")
        smoothed_trajectory = planner.smooth_joint_trajectory(joint_trajectory, smoothing_factor=0.25)
        
        # 7. 保存轨迹数据
        print(f"\n💾 保存轨迹数据")
        planner.save_trajectory_data(joint_trajectory, smoothed_trajectory, "integrated_trajectory")
        
        print(f"\n✅ 路径规划阶段完成！")
        
    except Exception as e:
        print(f"❌ 路径规划阶段出错: {e}")
        return
    
    # 第二部分：MuJoCo仿真
    print(f"\n🎬 第二部分：MuJoCo仿真")
    print("-" * 50)
    
    try:
        # 1. 初始化仿真器
        simulator = DualArmSimulator(xml_file)
        simulator.set_camera_params(distance=4.5, azimuth=-90, elevation=-30)
        
        # 2. 询问用户仿真选择
        print(f"\n🎮 仿真模式选择:")
        print("   1. 📼 播放原始轨迹")
        print("   2. 🌊 播放平滑轨迹 (推荐)")
        print("   3. 📊 对比播放 (先原始，后平滑)")
        print("   4. 🔄 循环播放平滑轨迹")
        print("   5. 👆 单步播放")
        print("   6. 🖼️  静态显示")
        
        choice = input("\n请选择仿真模式 (1-6): ").strip()
        
        if choice == "1":
            print(f"\n📼 播放原始轨迹...")
            simulator.animate_trajectory(joint_trajectory, dt=0.1, realtime=True, loop=False)
            
        elif choice == "2":
            print(f"\n🌊 播放平滑轨迹...")
            simulator.animate_trajectory(smoothed_trajectory, dt=0.1, realtime=True, loop=False)
            
        elif choice == "3":
            print(f"\n📊 对比播放模式...")
            print("   第一轮：原始轨迹")
            simulator.animate_trajectory(joint_trajectory, dt=0.1, realtime=True, loop=False)
            
            input("\n按回车键继续播放平滑轨迹...")
            print("   第二轮：平滑轨迹")
            simulator.animate_trajectory(smoothed_trajectory, dt=0.1, realtime=True, loop=False)
            
        elif choice == "4":
            print(f"\n🔄 循环播放平滑轨迹...")
            simulator.animate_trajectory(smoothed_trajectory, dt=0.1, realtime=True, loop=True)
            
        elif choice == "5":
            print(f"\n👆 单步播放模式...")
            simulator.step_by_step_animation(smoothed_trajectory, step_size=1)
            
        elif choice == "6":
            print(f"\n🖼️  静态显示模式...")
            frame_options = [0, len(smoothed_trajectory)//4, len(smoothed_trajectory)//2, 
                           3*len(smoothed_trajectory)//4, -1]
            frame_names = ["起始", "1/4处", "中点", "3/4处", "终点"]
            
            print("选择显示帧:")
            for i, (idx, name) in enumerate(zip(frame_options, frame_names)):
                print(f"   {i+1}. {name} (帧 {idx if idx >= 0 else len(smoothed_trajectory)+idx})")
            
            frame_choice = input("请选择帧 (1-5): ").strip()
            try:
                frame_idx = frame_options[int(frame_choice)-1]
                if frame_idx < 0:
                    frame_idx = len(smoothed_trajectory) + frame_idx
                simulator.static_display(smoothed_trajectory[frame_idx])
            except (ValueError, IndexError):
                print("❌ 无效选择，显示起始帧")
                simulator.static_display(smoothed_trajectory[0])
                
        else:
            print(f"❌ 无效选择，默认播放平滑轨迹")
            simulator.animate_trajectory(smoothed_trajectory, dt=0.1, realtime=True, loop=False)
        
        print(f"\n✅ 仿真阶段完成！")
        
    except Exception as e:
        print(f"❌ 仿真阶段出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 总结
    print(f"\n🎉 系统演示完成！")
    print("=" * 70)
    print("📋 生成的文件:")
    print("   • integrated_path_3d.png - 3D路径可视化")
    print("   • integrated_trajectory_raw.npy - 原始关节轨迹")
    print("   • integrated_trajectory_smooth.npy - 平滑关节轨迹")
    print("   • integrated_trajectory_summary.txt - 轨迹摘要信息")
    print("\n🔧 系统特点:")
    print("   ✓ 贝塞尔曲线路径规划")
    print("   ✓ 球面线性插值(SLERP)姿态插值")
    print("   ✓ 双臂约束逆运动学求解")
    print("   ✓ 轨迹平滑处理")
    print("   ✓ 实时MuJoCo仿真")
    print("   ✓ 模块化设计")
    print("=" * 70)


def quick_demo():
    """
    快速演示模式：自动执行完整流程
    """
    print("🚀 快速演示模式")
    print("   自动执行：路径规划 → 逆运动学 → 平滑 → 仿真")
    print("-" * 50)
    
    xml_file = 'dual_elfin15_scene_clean.xml'
    
    # 路径规划
    planner = BezierPathPlanner(xml_file)
    start_pose = planner.T_W_M_home.copy()
    end_pose = start_pose.copy()
    end_pose[:3, 3] += np.array([0.25, 0.15, 0.15])
    
    path_poses = planner.plan_bezier_path(start_pose, end_pose, num_points=25)
    joint_trajectory, _ = planner.solve_ik_for_path(path_poses)
    smoothed_trajectory = planner.smooth_joint_trajectory(joint_trajectory)
    
    # 直接仿真
    simulator = DualArmSimulator(xml_file)
    simulator.set_camera_params(distance=4.0, azimuth=-90, elevation=-35)
    simulator.animate_trajectory(smoothed_trajectory, dt=0.08, realtime=True, loop=False)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_demo()
    else:
        main() 