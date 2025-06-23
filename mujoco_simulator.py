#!/usr/bin/env python3
"""
MuJoCo双臂机器人仿真模块

该模块提供了MuJoCo仿真功能，用于：
1. 加载MuJoCo模型
2. 轨迹动画播放
3. 实时可视化
4. 相机控制

作者: frank
日期: 2024年6月19日
"""

import numpy as np
import time
from velocity_controller import VelocityController


class DualArmSimulator:
    """
    双臂MuJoCo仿真器类
    
    该类封装了MuJoCo仿真功能，用于可视化轨迹执行
    """
    
    def __init__(self, xml_file):
        """
        初始化仿真器
        
        Args:
            xml_file (str): MuJoCo XML文件路径
        """
        self.xml_file = xml_file
        self.model = None
        self.data = None
        self._load_model()
    
    def _load_model(self):
        """加载MuJoCo模型"""
        try:
            import mujoco
            self.model = mujoco.MjModel.from_xml_path(self.xml_file)
            self.data = mujoco.MjData(self.model)
            print("✅ MuJoCo模型加载成功")
            print(f"   模型文件: {self.xml_file}")
            print(f"   关节数量: {self.model.nq}")
            print(f"   执行器数量: {self.model.nu}")
        except ImportError:
            print("❌ 无法导入mujoco，仿真功能将不可用")
            print("   请确保已安装MuJoCo: pip install mujoco")
        except Exception as e:
            print(f"❌ 加载MuJoCo模型失败: {e}")
    
    def set_camera_params(self, distance=4.0, azimuth=-90, elevation=-45):
        """
        设置相机参数
        
        Args:
            distance (float): 相机距离：相机到观察目标的距离(米)，值越大视野越远
            azimuth (float): 方位角：水平旋转角度(度)，0°=正前方，90°=右侧，180°=正后方，270°=左侧
            elevation (float): 仰角：垂直角度(度)，正值=俯视，负值=仰视，0°=水平视角
        """
        self.camera_distance = distance
        self.camera_azimuth = azimuth
        self.camera_elevation = elevation
        
        print(f"📷 相机参数设置:")
        print(f"   距离: {distance}m")
        print(f"   方位角: {azimuth}° ({'正前方' if azimuth == 0 else '右侧' if azimuth == 90 else '正后方' if azimuth == 180 else '左侧' if azimuth == 270 else '自定义'})")
        print(f"   仰角: {elevation}° ({'俯视' if elevation > 0 else '仰视' if elevation < 0 else '水平'})")
    
    def animate_trajectory(self, joint_trajectory, control_mode='position', kp=10.0, dt=0.1, realtime=True, loop=False):
        """
        动画播放关节轨迹
        
        Args:
            joint_trajectory (list or np.ndarray): 关节角度轨迹
            control_mode (str): 控制模式, 'position' 或 'velocity_servo'
            kp (float): 速度伺服模式下的比例增益
            dt (float): 时间步长 (秒)
            realtime (bool): 是否实时播放
            loop (bool): 是否循环播放
        """
        if self.model is None:
            print("❌ MuJoCo模型未加载，无法进行仿真")
            return
            
        # 转换轨迹数据格式
        if isinstance(joint_trajectory, list):
            trajectory_array = np.array(joint_trajectory)
        else:
            trajectory_array = joint_trajectory
            
        if len(trajectory_array.shape) != 2 or trajectory_array.shape[1] != self.model.nq:
            print(f"❌ 轨迹数据格式错误，期望形状为 (N, {self.model.nq})，实际为 {trajectory_array.shape}")
            return
            
        try:
            import mujoco
            import mujoco.viewer
            
            print(f"🎬 开始播放轨迹动画")
            print(f"   控制模式: {'直接位置控制' if control_mode == 'position' else '速度伺服控制'}")
            if control_mode == 'velocity_servo':
                print(f"   P 控制增益 (Kp): {kp}")

            print(f"   轨迹点数: {len(trajectory_array)}")
            print(f"   播放速度: {'实时' if realtime else '快速'}")
            print(f"   时间步长: {dt}s")
            print(f"   循环播放: {'是' if loop else '否'}")
            print("\n📖 控制说明:")
            print("   空格键: 暂停/继续")
            print("   ESC键: 退出仿真")
            print("   鼠标: 旋转视角")
            print("   滚轮: 缩放")
            
            # 初始化速度控制器
            if control_mode == 'velocity_servo':
                controller = VelocityController(kp=kp)
                # 计算每个轨迹点需要仿真的步数
                n_steps = int(dt / self.model.opt.timestep)
                print(f"   每个轨迹点将仿真 {n_steps} 步 (仿真步长: {self.model.opt.timestep * 1000:.2f} ms)")

            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 设置相机位置和视角参数
                viewer.cam.distance = self.camera_distance
                viewer.cam.azimuth = self.camera_azimuth
                viewer.cam.elevation = self.camera_elevation
                
                frame_idx = 0
                play_count = 0
                
                start_time = time.time()
                sim_time = 0

                while viewer.is_running():
                    
                    if frame_idx < len(trajectory_array):
                        target_qpos = trajectory_array[frame_idx]

                        if control_mode == 'position':
                            # --- 模式1: 直接设置关节位置 (运动学) ---
                            self.data.qpos[:self.model.nq] = target_qpos
                            mujoco.mj_forward(self.model, self.data)
                            viewer.sync()
                            if realtime:
                                time.sleep(dt)

                        elif control_mode == 'velocity_servo':
                            # --- 模式2: 速度伺服控制 (动力学) ---
                            # 在 dt 时间内通过P控制器跟踪目标位置
                            for _ in range(n_steps):
                                current_qpos = self.data.qpos[:self.model.nq]
                                vel_command = controller.compute_velocity(target_qpos, current_qpos)
                                
                                # 将速度指令发送给执行器
                                self.data.ctrl[:self.model.nu] = vel_command
                                
                                # 执行一步仿真
                                mujoco.mj_step(self.model, self.data)
                            
                            # 更新显示
                            viewer.sync()
                            
                            # 实时播放控制
                            if realtime:
                                sim_time += dt
                                elapsed_time = time.time() - start_time
                                sleep_time = sim_time - elapsed_time
                                if sleep_time > 0:
                                    time.sleep(sleep_time)

                        # 更新进度
                        if frame_idx % 10 == 0 or frame_idx == len(trajectory_array) - 1:
                            progress = ((frame_idx + 1) / len(trajectory_array)) * 100
                            print(f"🎮 播放进度: {frame_idx + 1}/{len(trajectory_array)} ({progress:.1f}%)")
                        
                        frame_idx += 1
                        
                    else:
                        # 轨迹播放完成
                        if loop:
                            frame_idx = 0  # 重新开始
                            play_count += 1
                            print(f"🔄 第 {play_count + 1} 次循环播放")
                            start_time = time.time()
                            sim_time = 0
                        else:
                            # 保持最后一帧
                            if control_mode == 'velocity_servo':
                                # 清零速度，防止漂移
                                self.data.ctrl[:self.model.nu] = 0
                                for _ in range(100): # 稳定一下
                                    mujoco.mj_step(self.model, self.data)
                                viewer.sync()

                            print("✅ 轨迹播放完成，按ESC退出")
                            while viewer.is_running():
                                viewer.sync()
                                time.sleep(0.01)
                            break
                
        except ImportError:
            print("❌ 无法导入mujoco.viewer，使用备用显示方案")
            self._print_trajectory_summary(trajectory_array)
        except Exception as e:
            print(f"❌ 仿真过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    def step_by_step_animation(self, joint_trajectory, step_size=1):
        """
        单步播放轨迹动画
        
        Args:
            joint_trajectory (list or np.ndarray): 关节角度轨迹
            step_size (int): 每次前进的步数
        """
        if self.model is None:
            print("❌ MuJoCo模型未加载，无法进行仿真")
            return
            
        # 转换轨迹数据格式
        if isinstance(joint_trajectory, list):
            trajectory_array = np.array(joint_trajectory)
        else:
            trajectory_array = joint_trajectory
            
        try:
            import mujoco
            import mujoco.viewer
            
            print(f"🎬 单步播放模式")
            print(f"   轨迹点数: {len(trajectory_array)}")
            print(f"   步长: {step_size}")
            print("\n📖 控制说明:")
            print("   空格键: 前进一步")
            print("   ESC键: 退出仿真")
            print("   鼠标: 旋转视角")
            
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 设置相机参数
                viewer.cam.distance = self.camera_distance
                viewer.cam.azimuth = self.camera_azimuth
                viewer.cam.elevation = self.camera_elevation
                
                frame_idx = 0
                
                while viewer.is_running() and frame_idx < len(trajectory_array):
                    # 设置关节角度
                    self.data.qpos[:12] = trajectory_array[frame_idx]
                    
                    # 前向动力学
                    mujoco.mj_forward(self.model, self.data)
                    
                    # 更新显示
                    viewer.sync()
                    
                    print(f"🎮 当前帧: {frame_idx + 1}/{len(trajectory_array)}")
                    print("   按空格键继续...")
                    
                    # 等待用户输入
                    input()
                    
                    frame_idx += step_size
                
                print("✅ 单步播放完成")
                
        except ImportError:
            print("❌ 无法导入mujoco.viewer")
        except Exception as e:
            print(f"❌ 单步仿真过程中发生错误: {e}")
    
    def static_display(self, joint_angles):
        """
        静态显示单个关节配置
        
        Args:
            joint_angles (np.ndarray): 12个关节角度
        """
        if self.model is None:
            print("❌ MuJoCo模型未加载，无法进行仿真")
            return
            
        try:
            import mujoco
            import mujoco.viewer
            
            print(f"🖼️  静态显示模式")
            print(f"   关节角度: {np.rad2deg(joint_angles)}")
            print("\n📖 控制说明:")
            print("   ESC键: 退出显示")
            print("   鼠标: 旋转视角")
            
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 设置相机参数
                viewer.cam.distance = self.camera_distance
                viewer.cam.azimuth = self.camera_azimuth
                viewer.cam.elevation = self.camera_elevation
                
                # 设置关节角度
                self.data.qpos[:12] = joint_angles
                
                # 前向动力学
                mujoco.mj_forward(self.model, self.data)
                
                # 保持显示
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.01)
                
        except ImportError:
            print("❌ 无法导入mujoco.viewer")
        except Exception as e:
            print(f"❌ 静态显示过程中发生错误: {e}")
    
    def _print_trajectory_summary(self, trajectory_array):
        """打印轨迹摘要信息（备用方案）"""
        print("\n" + "=" * 50)
        print("📊 轨迹摘要信息")
        print("=" * 50)
        print(f"总帧数: {len(trajectory_array)}")
        print(f"关节数量: {trajectory_array.shape[1]}")
        
        if trajectory_array.size > 0:
            # 分析关节角度变化范围
            print("\n🔧 左臂关节角度范围 (度):")
            for i in range(6):
                min_angle = np.rad2deg(np.min(trajectory_array[:, i]))
                max_angle = np.rad2deg(np.max(trajectory_array[:, i]))
                range_deg = max_angle - min_angle
                print(f"  关节 {i+1}: [{min_angle:6.1f}, {max_angle:6.1f}] (范围: {range_deg:5.1f}°)")
            
            print("\n🔧 右臂关节角度范围 (度):")
            for i in range(6, 12):
                min_angle = np.rad2deg(np.min(trajectory_array[:, i]))
                max_angle = np.rad2deg(np.max(trajectory_array[:, i]))
                range_deg = max_angle - min_angle
                print(f"  关节 {i-5}: [{min_angle:6.1f}, {max_angle:6.1f}] (范围: {range_deg:5.1f}°)")
            
            # 计算轨迹平滑度
            if len(trajectory_array) > 1:
                velocity = np.diff(trajectory_array, axis=0)
                acceleration = np.diff(velocity, axis=0)
                
                avg_velocity = np.mean(np.abs(velocity))
                avg_acceleration = np.mean(np.abs(acceleration))
                
                print(f"\n📈 轨迹统计:")
                print(f"  平均角速度: {avg_velocity:.4f} rad/step")
                print(f"  平均角加速度: {avg_acceleration:.4f} rad/step²")


def load_trajectory_from_file(filename):
    """
    从文件加载轨迹数据
    
    Args:
        filename (str): 轨迹文件路径
        
    Returns:
        np.ndarray: 轨迹数据
    """
    try:
        trajectory = np.load(filename)
        print(f"✅ 轨迹文件加载成功: {filename}")
        print(f"   轨迹形状: {trajectory.shape}")
        return trajectory
    except Exception as e:
        print(f"❌ 加载轨迹文件失败: {e}")
        return None


def demo_simulation():
    """
    演示仿真功能
    """
    print("=" * 60)
    print("MuJoCo双臂机器人仿真演示")
    print("=" * 60)
    
    # XML文件路径
    xml_file = 'dual_elfin15_scene_clean.xml'
    
    # 创建仿真器
    simulator = DualArmSimulator(xml_file)
    simulator.set_camera_params(distance=4.0, azimuth=-90, elevation=-45)
    
    # 尝试加载现有的轨迹文件
    trajectory_files = [
        'bezier_trajectory_smooth.npy',
        'smoothed_trajectory.npy',
        'joint_trajectory.npy'
    ]
    
    trajectory = None
    for filename in trajectory_files:
        trajectory = load_trajectory_from_file(filename)
        if trajectory is not None:
            break
    
    if trajectory is not None:
        print(f"\n🎬 开始仿真演示...")
        
        # 选择仿真模式
        print("\n请选择仿真模式:")
        print("1. 实时播放")
        print("2. 快速播放")
        print("3. 循环播放")
        print("4. 单步播放")
        print("5. 静态显示（仅第一帧）")
        
        choice = input("请输入选择 (1-5): ").strip()
        
        if choice == "1":
            simulator.animate_trajectory(trajectory, dt=0.1, realtime=True, loop=False)
        elif choice == "2":
            simulator.animate_trajectory(trajectory, dt=0.05, realtime=False, loop=False)
        elif choice == "3":
            simulator.animate_trajectory(trajectory, dt=0.1, realtime=True, loop=True)
        elif choice == "4":
            simulator.step_by_step_animation(trajectory, step_size=1)
        elif choice == "5":
            simulator.static_display(trajectory[0])
        else:
            print("❌ 无效选择，使用默认实时播放")
            simulator.animate_trajectory(trajectory, dt=0.1, realtime=True, loop=False)
    else:
        print("❌ 未找到轨迹文件，请先运行路径规划生成轨迹")


if __name__ == '__main__':
    demo_simulation() 