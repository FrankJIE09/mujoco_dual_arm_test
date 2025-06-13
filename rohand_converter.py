#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoHand灵巧手简化转换脚本
将25个关节简化为6个主要自由度：
- 4个手指弯曲自由度（食指、中指、无名指、小指）
- 1个拇指旋转自由度
- 1个拇指弯曲自由度
"""

import mujoco
import os
import sys
import shutil
import re
import xml.etree.ElementTree as ET


class RoHandSimplifiedConverter:
    """RoHand灵巧手简化转换器"""

    def __init__(self):
        # 定义主要控制关节映射
        self.main_joints = {
            # 手指弯曲关节（每个手指选择一个主要关节）
            'if_bend': 'if_proximal_link',  # 食指弯曲
            'mf_bend': 'mf_proximal_link',  # 中指弯曲
            'rf_bend': 'rf_proximal_link',  # 无名指弯曲
            'lf_bend': 'lf_proximal_link',  # 小指弯曲
            # 拇指关节
            'th_rotate': 'th_root_link',  # 拇指旋转
            'th_bend': 'th_proximal_link',  # 拇指弯曲
        }

        # 定义从动关节及其耦合比例
        self.coupled_joints = {
            # 食指联动
            'if_distal_link': ('if_bend', 0.8),  # 远端跟随近端 80%
            'if_connecting_link': ('if_bend', 0.3),  # 连接关节跟随 30%
            'if_slider_abpart_link': ('if_bend', 0.2),  # 滑动块跟随 20%

            # 中指联动
            'mf_distal_link': ('mf_bend', 0.8),
            'mf_connecting_link': ('mf_bend', 0.3),
            'mf_slider_abpart_link': ('mf_bend', 0.2),

            # 无名指联动
            'rf_distal_link': ('rf_bend', 0.8),
            'rf_connecting_link': ('rf_bend', 0.3),
            'rf_slider_abpart_link': ('rf_bend', 0.2),

            # 小指联动
            'lf_distal_link': ('lf_bend', 0.8),
            'lf_connecting_link': ('lf_bend', 0.3),
            'lf_slider_abpart_link': ('lf_bend', 0.2),

            # 拇指联动
            'th_distal_link': ('th_bend', 0.8),
            'th_connecting_link': ('th_bend', 0.3),
        }

    def convert_rohand(self, hand_type="left"):
        """
        转换RoHand灵巧手为简化版本

        Args:
            hand_type: "left" 或 "right"
        """
        print(f"🤖 开始转换RoHand {hand_type}手为简化版本...")

        # 路径定义
        project_root = os.getcwd()
        urdf_path = os.path.join(project_root, 'rohand_urdf_ros2', 'urdf', f'rohand_{hand_type}.urdf')

        if hand_type == 'left':
            mesh_source_dir = os.path.join(project_root, 'rohand_urdf_ros2', 'meshes_l')
        else:
            mesh_source_dir = os.path.join(project_root, 'rohand_urdf_ros2', 'meshes_r')

        output_dir = os.path.join(project_root, f'mjcf_models/rohand_{hand_type}_simplified')

        if not os.path.exists(urdf_path):
            print(f"❌ URDF文件不存在: {urdf_path}")
            return False

        if not os.path.exists(mesh_source_dir):
            print(f"❌ Mesh源目录不存在: {mesh_source_dir}")
            return False

        # 创建输出目录
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # 复制mesh文件
        print("📁 复制mesh文件...")
        copied_files = 0
        for root, _, files in os.walk(mesh_source_dir):
            for file in files:
                if file.lower().endswith(('.stl', '.dae', '.obj', '.ply')):
                    shutil.copy(os.path.join(root, file), output_dir)
                    copied_files += 1
        print(f"📁 复制了 {copied_files} 个mesh文件")

        # 处理URDF
        print("🔧 处理URDF文件...")
        modified_urdf_path = self.process_urdf(urdf_path, output_dir, hand_type)

        # 转换为MJCF
        print("🔄 转换为MJCF...")
        mjcf_path = self.convert_to_mjcf(modified_urdf_path, output_dir, hand_type)

        if mjcf_path:
            # 添加约束和actuator
            print("⚙️ 添加约束和actuator...")
            self.add_constraints_and_actuators(mjcf_path, hand_type)
            print(f"✅ 转换完成: {mjcf_path}")
            return mjcf_path
        else:
            print("❌ 转换失败")
            return False

    def process_urdf(self, urdf_path, output_dir, hand_type):
        """处理URDF文件，修改mesh路径"""
        with open(urdf_path, 'r', encoding='utf-8') as f:
            urdf_content = f.read()

        # 修改mesh路径
        def replace_mesh_path(match):
            filename = os.path.basename(match.group(1))
            return f'filename="{filename}"'

        pattern = r'filename="package://[^"]+/([^"]+)"'
        modified_urdf_content = re.sub(pattern, replace_mesh_path, urdf_content)

        # 保存修改后的URDF
        modified_urdf_path = os.path.join(output_dir, f'rohand_{hand_type}_simplified.urdf')
        with open(modified_urdf_path, 'w', encoding='utf-8') as f:
            f.write(modified_urdf_content)

        return modified_urdf_path

    def convert_to_mjcf(self, urdf_path, output_dir, hand_type):
        """将URDF转换为MJCF"""
        original_cwd = os.getcwd()
        try:
            # 进入输出目录
            os.chdir(output_dir)

            # 加载URDF
            model = mujoco.MjModel.from_xml_path(os.path.basename(urdf_path))
            print("✅ URDF加载成功!")

            # 保存MJCF
            output_mjcf_name = f"rohand_{hand_type}_simplified.xml"
            mujoco.mj_saveLastXML(output_mjcf_name, model)
            print(f"✅ MJCF文件已生成: {output_mjcf_name}")

            return os.path.join(output_dir, output_mjcf_name)

        except Exception as e:
            print(f"❌ 转换失败: {e}")
            return None
        finally:
            os.chdir(original_cwd)

    def add_constraints_and_actuators(self, mjcf_path, hand_type):
        """添加约束和actuator到MJCF文件"""
        try:
            # 解析MJCF文件
            tree = ET.parse(mjcf_path)
            root = tree.getroot()

            # 修改模型名称
            root.set('model', f'rohand_{hand_type}_simplified')

            # 创建或获取actuator section
            actuator_section = root.find('actuator')
            if actuator_section is None:
                actuator_section = ET.SubElement(root, 'actuator')
            else:
                # 清空现有actuator
                actuator_section.clear()

            # 创建或获取equality section（用于关节约束）
            equality_section = root.find('equality')
            if equality_section is None:
                equality_section = ET.SubElement(root, 'equality')
            else:
                equality_section.clear()

            # 添加主要控制关节的actuator
            for control_name, joint_name in self.main_joints.items():
                actuator = ET.SubElement(actuator_section, 'general')
                actuator.set('name', f'{control_name}_actuator')
                actuator.set('joint', joint_name)
                actuator.set('gainprm', '10')

                # 根据关节类型设置控制范围
                if 'bend' in control_name:
                    actuator.set('ctrlrange', '-1.57 1.57')  # ±90度
                elif 'rotate' in control_name:
                    actuator.set('ctrlrange', '-0.6 1.6')  # 根据拇指实际范围

            # 添加关节耦合约束
            for slave_joint, (master_control, ratio) in self.coupled_joints.items():
                master_joint = self.main_joints[master_control]

                # 创建关节约束
                constraint = ET.SubElement(equality_section, 'joint')
                constraint.set('name', f'{slave_joint}_coupling')
                constraint.set('joint1', master_joint)
                constraint.set('joint2', slave_joint)
                constraint.set('polycoef', f'0 {ratio} 0 0 0')  # 线性耦合关系

            # 创建默认keyframe
            keyframe_section = root.find('keyframe')
            if keyframe_section is None:
                keyframe_section = ET.SubElement(root, 'keyframe')
            else:
                keyframe_section.clear()

            # 添加默认姿态（需要包含所有25个关节的值）
            default_key = ET.SubElement(keyframe_section, 'key')
            default_key.set('name', 'home')
            # 25个关节的初始位置：所有关节都设为0（张开状态）
            default_qpos = ' '.join(['0'] * 25)
            default_key.set('qpos', default_qpos)

            # 添加一个握拳姿态
            fist_key = ET.SubElement(keyframe_section, 'key')
            fist_key.set('name', 'fist')
            fist_qpos = self.generate_pose_qpos('fist')
            fist_key.set('qpos', fist_qpos)

            # 添加一个OK手势
            ok_key = ET.SubElement(keyframe_section, 'key')
            ok_key.set('name', 'ok')
            ok_qpos = self.generate_pose_qpos('ok')
            ok_key.set('qpos', ok_qpos)

            # 美化XML格式
            self.indent_xml(root)

            # 保存修改后的MJCF
            tree.write(mjcf_path, encoding='utf-8', xml_declaration=True)
            print(f"✅ 已添加 {len(self.main_joints)} 个主要actuator和约束")

            return True

        except Exception as e:
            print(f"❌ 添加约束和actuator失败: {e}")
            return False

    def generate_pose_qpos(self, pose_type):
        """
        生成包含所有25个关节值的姿态
        
        Args:
            pose_type: 'fist', 'ok' 等预设姿态类型
        """
        # 定义关节顺序（按照URDF中的顺序）
        joint_order = [
            'if_slider_link', 'if_slider_abpart_link', 'if_proximal_link', 
            'if_distal_link', 'if_connecting_link',
            'mf_slider_link', 'mf_slider_abpart_link', 'mf_proximal_link',
            'mf_distal_link', 'mf_connecting_link',
            'rf_slider_link', 'rf_slider_abpart_link', 'rf_proximal_link',
            'rf_distal_link', 'rf_connecting_link',
            'lf_slider_link', 'lf_slider_abpart_link', 'lf_proximal_link',
            'lf_distal_link', 'lf_connecting_link',
            'th_root_link', 'th_proximal_link', 'th_slider_link',
            'th_connecting_link', 'th_distal_link'
        ]
        
        # 初始化所有关节为0
        joint_values = {joint: 0.0 for joint in joint_order}
        
        if pose_type == 'fist':
            # 握拳姿态：所有手指弯曲
            # 主要关节
            joint_values['if_proximal_link'] = 1.2
            joint_values['mf_proximal_link'] = 1.2
            joint_values['rf_proximal_link'] = 1.2
            joint_values['lf_proximal_link'] = 1.2
            joint_values['th_proximal_link'] = 1.2
            
            # 从动关节按照耦合比例
            joint_values['if_distal_link'] = 1.2 * 0.8
            joint_values['if_connecting_link'] = 1.2 * 0.3
            joint_values['if_slider_abpart_link'] = 1.2 * 0.2
            
            joint_values['mf_distal_link'] = 1.2 * 0.8
            joint_values['mf_connecting_link'] = 1.2 * 0.3
            joint_values['mf_slider_abpart_link'] = 1.2 * 0.2
            
            joint_values['rf_distal_link'] = 1.2 * 0.8
            joint_values['rf_connecting_link'] = 1.2 * 0.3
            joint_values['rf_slider_abpart_link'] = 1.2 * 0.2
            
            joint_values['lf_distal_link'] = 1.2 * 0.8
            joint_values['lf_connecting_link'] = 1.2 * 0.3
            joint_values['lf_slider_abpart_link'] = 1.2 * 0.2
            
            joint_values['th_distal_link'] = 1.2 * 0.8
            joint_values['th_connecting_link'] = 1.2 * 0.3
            
        elif pose_type == 'ok':
            # OK手势：食指和拇指接触
            joint_values['if_proximal_link'] = 1.0
            joint_values['th_root_link'] = 0.5
            joint_values['th_proximal_link'] = 1.0
            
            # 从动关节
            joint_values['if_distal_link'] = 1.0 * 0.8
            joint_values['if_connecting_link'] = 1.0 * 0.3
            joint_values['if_slider_abpart_link'] = 1.0 * 0.2
            
            joint_values['th_distal_link'] = 1.0 * 0.8
            joint_values['th_connecting_link'] = 1.0 * 0.3
        
        # 按照关节顺序生成qpos字符串
        qpos_values = [str(joint_values[joint]) for joint in joint_order]
        return ' '.join(qpos_values)

    def indent_xml(self, elem, level=0):
        """美化XML格式"""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self.indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


def main():
    """主函数"""
    print("🤖 RoHand灵巧手简化转换工具")
    print("=" * 50)
    print("将25个关节简化为6个主要自由度：")
    print("- 4个手指弯曲（食指、中指、无名指、小指）")
    print("- 1个拇指旋转")
    print("- 1个拇指弯曲")
    print("=" * 50)

    converter = RoHandSimplifiedConverter()

    success_count = 0
    models = []

    # 转换左手
    print("\n🫲 转换左手模型...")
    left_model = converter.convert_rohand("left")
    if left_model:
        success_count += 1
        models.append(left_model)

    # 转换右手
    print("\n🫱 转换右手模型...")
    right_model = converter.convert_rohand("right")
    if right_model:
        success_count += 1
        models.append(right_model)

    print("\n" + "=" * 50)
    print(f"✅ 转换完成: {success_count}/2 成功")

    if models:
        print("\n📂 生成的简化模型:")
        for model in models:
            print(f"  {os.path.relpath(model, os.getcwd())}")

        print("\n🎮 控制自由度说明:")
        print("  0: if_bend  - 食指弯曲")
        print("  1: mf_bend  - 中指弯曲")
        print("  2: rf_bend  - 无名指弯曲")
        print("  3: lf_bend  - 小指弯曲")
        print("  4: th_rotate - 拇指旋转")
        print("  5: th_bend  - 拇指弯曲")

        print("\n🚀 使用以下命令查看简化模型:")
        for model in models:
            print(f"python mjcf_viewer.py {os.path.relpath(model, os.getcwd())}")

        print("\n🔧 预设姿态:")
        print("  - home: 张开状态")
        print("  - fist: 握拳状态")
        print("  - ok: OK手势")


if __name__ == '__main__':
    main()