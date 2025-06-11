#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MJCF模型查看器
支持加载MJCF文件并提供交互式关节控制

依赖:
- mujoco
- numpy
- matplotlib (可选，用于数据可视化)
"""

import mujoco
import mujoco.viewer
import numpy as np
import argparse
import os
import sys
import time
import threading
from pathlib import Path


class MJCFViewer:
    """MJCF模型查看器类"""

    def __init__(self, mjcf_path):
        """
        初始化查看器

        Args:
            mjcf_path (str): MJCF文件路径
        """
        self.mjcf_path = mjcf_path
        self.model = None
        self.data = None
        self.viewer = None
        self.running = False

        # 关节控制参数
        self.joint_targets = {}
        self.joint_velocities = {}
        self.control_mode = 'position'  # 'position' 或 'velocity'

        print(f"🤖 初始化MJCF查看器")
        print(f"📁 文件路径: {mjcf_path}")

    def load_model(self):
        """加载MJCF模型"""
        try:
            if not os.path.exists(self.mjcf_path):
                raise FileNotFoundError(f"MJCF文件不存在: {self.mjcf_path}")

            print("🔄 正在加载MJCF模型...")
            self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
            self.data = mujoco.MjData(self.model)

            print("✅ 模型加载成功!")
            self.print_model_info()

            # 初始化关节目标位置
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name:
                    self.joint_targets[joint_name] = 0.0
                    self.joint_velocities[joint_name] = 0.0

        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            sys.exit(1)

    def print_model_info(self):
        """打印模型信息"""
        print("\n📊 模型信息:")
        print(f"  🔗 刚体数量: {self.model.nbody}")
        print(f"  🔄 关节数量: {self.model.njnt}")
        print(f"  ⚙️  自由度: {self.model.nv}")
        print(f"  🎮 执行器数量: {self.model.nu}")

        # 打印关节信息
        if self.model.njnt > 0:
            print(f"\n🔄 关节列表:")
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                joint_type = self.model.jnt_type[i]
                joint_range = self.model.jnt_range[i]

                # 关节类型映射
                type_map = {
                    0: "自由",
                    1: "球形",
                    2: "滑动",
                    3: "铰链",
                    4: "螺旋"
                }

                type_name = type_map.get(joint_type, f"未知({joint_type})")

                if joint_name:
                    if joint_range[0] == joint_range[1]:  # 无限制
                        range_str = "无限制"
                    else:
                        range_str = f"[{joint_range[0]:.2f}, {joint_range[1]:.2f}]"

                    print(f"    {i + 1:2d}. {joint_name:<15} 类型: {type_name:<6} 范围: {range_str}")

        # 打印执行器信息
        if self.model.nu > 0:
            print(f"\n⚙️  执行器列表:")
            for i in range(self.model.nu):
                actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if actuator_name:
                    print(f"    {i + 1:2d}. {actuator_name}")

    def start_viewer(self):
        """启动可视化查看器"""
        print("\n🚀 启动可视化查看器...")
        print("\n📝 控制说明:")
        print("  🖱️  鼠标左键拖拽: 旋转视角")
        print("  🖱️  鼠标右键拖拽: 平移视角")
        print("  🖱️  鼠标滚轮: 缩放")
        print("  ⌨️  空格键: 暂停/继续仿真")
        print("  ⌨️  Ctrl+R: 重置仿真")
        print("  ⌨️  Tab: 显示/隐藏帮助")
        print("  ⌨️  ESC: 退出程序")
        print("\n🎮 关节控制:")
        print("  在终端中输入命令来控制关节 (另开终端窗口)")
        print("  命令格式: <关节名称> <目标位置>")
        print("  例如: joint1 1.57")
        print("  输入 'help' 查看所有命令")
        print("  输入 'quit' 退出程序")

        self.running = True

        # 启动控制线程
        control_thread = threading.Thread(target=self.control_loop, daemon=True)
        control_thread.start()

        # 启动查看器
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer

            while self.running and viewer.is_running():
                self.data.qvel = 10 * (self.data.ctrl-self.data.qpos )
                # 更新仿真
                mujoco.mj_step(self.model, self.data)
                # 同步查看器
                viewer.sync()
                # 控制更新频率
                time.sleep(0.01)

    def control_loop(self):
        """关节控制循环"""
        print(f"\n💡 关节控制已启动 (在当前终端输入命令)")
        self.print_control_help()

        while self.running:
            try:
                command = input("🎮 > ").strip()
                if not command:
                    continue

                if command.lower() == 'quit':
                    self.running = False
                    break
                elif command.lower() == 'help':
                    self.print_control_help()
                elif command.lower() == 'list':
                    self.list_joints()
                elif command.lower() == 'reset':
                    self.reset_joints()
                elif command.lower() == 'info':
                    self.print_current_state()
                elif command.lower().startswith('mode'):
                    self.set_control_mode(command)
                else:
                    self.parse_joint_command(command)

            except (EOFError, KeyboardInterrupt):
                self.running = False
                break
            except Exception as e:
                print(f"❌ 命令错误: {e}")

    def print_control_help(self):
        """打印控制帮助"""
        print("\n🎮 关节控制命令:")
        print("  <关节名> <值>     - 设置关节目标位置/速度")
        print("  list             - 列出所有关节")
        print("  reset            - 重置所有关节到零位")
        print("  info             - 显示当前关节状态")
        print("  mode <position|velocity> - 切换控制模式")
        print("  help             - 显示此帮助")
        print("  quit             - 退出程序")
        print(f"\n当前控制模式: {self.control_mode}")

    def list_joints(self):
        """列出所有关节"""
        print(f"\n🔄 关节列表 (共{self.model.njnt}个):")
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                current_pos = self.data.qpos[i] if i < len(self.data.qpos) else 0.0
                current_vel = self.data.qvel[i] if i < len(self.data.qvel) else 0.0
                target = self.joint_targets.get(joint_name, 0.0)

                joint_range = self.model.jnt_range[i]
                if joint_range[0] == joint_range[1]:
                    range_str = "无限制"
                else:
                    range_str = f"[{joint_range[0]:.2f}, {joint_range[1]:.2f}]"

                print(
                    f"  {i + 1:2d}. {joint_name:<15} 位置: {current_pos:8.3f} 速度: {current_vel:8.3f} 目标: {target:8.3f} 范围: {range_str}")

    def reset_joints(self):
        """重置所有关节"""
        print("🔄 重置所有关节...")
        for joint_name in self.joint_targets:
            self.joint_targets[joint_name] = 0.0
            self.joint_velocities[joint_name] = 0.0

        # 重置仿真状态
        mujoco.mj_resetData(self.model, self.data)
        print("✅ 关节已重置")

    def print_current_state(self):
        """打印当前状态"""
        print(f"\n📊 当前状态:")
        print(f"  ⏰ 仿真时间: {self.data.time:.3f}s")
        print(f"  🎮 控制模式: {self.control_mode}")
        print(f"  🔄 关节数量: {self.model.njnt}")

        if self.model.njnt > 0:
            print(f"  📈 关节状态摘要:")
            positions = self.data.qpos[:self.model.njnt]
            velocities = self.data.qvel[:self.model.njnt] if self.model.njnt <= len(self.data.qvel) else [
                                                                                                             0.0] * self.model.njnt

            print(f"    位置范围: [{np.min(positions):.3f}, {np.max(positions):.3f}]")
            print(f"    速度范围: [{np.min(velocities):.3f}, {np.max(velocities):.3f}]")

    def set_control_mode(self, command):
        """设置控制模式"""
        parts = command.split()
        if len(parts) != 2:
            print("❌ 用法: mode <position|velocity>")
            return

        mode = parts[1].lower()
        if mode in ['position', 'velocity']:
            self.control_mode = mode
            print(f"✅ 控制模式已切换到: {mode}")
        else:
            print("❌ 无效模式，请使用 'position' 或 'velocity'")

    def parse_joint_command(self, command):
        """解析关节控制命令"""
        parts = command.split()
        if len(parts) != 2:
            print("❌ 用法: <关节名称> <数值>")
            return

        joint_name, value_str = parts

        try:
            value = float(value_str)
        except ValueError:
            print(f"❌ 无效数值: {value_str}")
            return

        # 查找关节ID
        joint_id = None
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name == joint_name:
                joint_id = i
                break

        if joint_id is None:
            print(f"❌ 未找到关节: {joint_name}")
            print("💡 使用 'list' 命令查看可用关节")
            return

        # 检查关节范围
        joint_range = self.model.jnt_range[joint_id]
        if joint_range[0] != joint_range[1]:  # 有限制
            if value < joint_range[0] or value > joint_range[1]:
                print(f"⚠️  值 {value:.3f} 超出关节范围 [{joint_range[0]:.3f}, {joint_range[1]:.3f}]")
                print("是否继续? (y/N):")
                response = input().strip().lower()
                if response != 'y' and response != 'yes':
                    return

        # 设置目标值
        if self.control_mode == 'position':
            self.joint_targets[joint_name] = value
            # 直接设置关节位置（简单控制）
            if joint_id < len(self.data.qpos):
                self.data.qpos[joint_id] = value
            print(f"✅ 设置关节 {joint_name} 位置目标: {value:.3f}")
        else:  # velocity mode
            self.joint_velocities[joint_name] = value
            if joint_id < len(self.data.qvel):
                self.data.qvel[joint_id] = value
            print(f"✅ 设置关节 {joint_name} 速度目标: {value:.3f}")


def find_mjcf_files(directory="mjcf_models"):
    """查找可用的MJCF文件"""
    if not os.path.exists(directory):
        return []

    mjcf_files = []
    # 递归搜索所有子目录中的xml文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.xml'):
                mjcf_files.append(os.path.join(root, file))

    return sorted(mjcf_files)


def list_available_models():
    """列出可用的模型"""
    mjcf_files = find_mjcf_files()

    if not mjcf_files:
        print("❌ 未找到MJCF文件")
        print("💡 请先运行 ./urdf_to_mjcf.sh 转换URDF文件")
        return None

    print("📁 可用的MJCF模型:")
    for i, file in enumerate(mjcf_files, 1):
        basename = os.path.basename(file)
        size = os.path.getsize(file)
        size_str = f"{size / 1024:.1f}KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f}MB"
        print(f"  {i:2d}. {basename:<20} ({size_str})")

    return mjcf_files


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MJCF模型查看器")
    parser.add_argument("model", nargs="?", help="MJCF文件路径或模型名称")
    parser.add_argument("--list", "-l", action="store_true", help="列出可用模型")
    parser.add_argument("--directory", "-d", default="mjcf_models", help="MJCF文件目录")

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    mjcf_path = None

    if args.model:
        # 检查是否为完整路径
        if os.path.exists(args.model):
            mjcf_path = args.model
        else:
            # 在指定目录中查找
            potential_path = os.path.join(args.directory, args.model)
            if os.path.exists(potential_path):
                mjcf_path = potential_path
            elif os.path.exists(potential_path + ".xml"):
                mjcf_path = potential_path + ".xml"
            else:
                print(f"❌ 未找到模型文件: {args.model}")
                mjcf_files = list_available_models()
                return
    else:
        # 交互式选择
        mjcf_files = list_available_models()
        if not mjcf_files:
            return

        try:
            choice = input(f"\n请选择模型 (1-{len(mjcf_files)}): ").strip()
            index = int(choice) - 1
            if 0 <= index < len(mjcf_files):
                mjcf_path = mjcf_files[index]
            else:
                print("❌ 无效选择")
                return
        except (ValueError, KeyboardInterrupt):
            print("❌ 操作取消")
            return

    if not mjcf_path:
        print("❌ 未指定模型文件")
        return

    # 启动查看器
    try:
        viewer = MJCFViewer(mjcf_path)
        viewer.load_model()
        viewer.start_viewer()
    except KeyboardInterrupt:
        print("\n\n👋 程序已退出")
    except Exception as e:
        print(f"❌ 程序错误: {e}")


if __name__ == "__main__":
    main()
