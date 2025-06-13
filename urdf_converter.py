#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可靠的URDF转MJCF批量转换脚本
彻底解决路径问题
"""

import mujoco
import os
import sys
import shutil
import re
import xml.etree.ElementTree as ET


def add_actor_to_mjcf(mjcf_path, robot_name):
    """
    在生成的MJCF文件中添加actor元素
    """
    try:
        # 解析MJCF文件
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        
        # 查找worldbody元素
        worldbody = root.find('worldbody')
        if worldbody is None:
            print("❌ 未找到worldbody元素")
            return False
            
        # 查找主要的robot body (通常是第一个body元素)
        robot_body = worldbody.find('body')
        if robot_body is None:
            print("❌ 未找到robot body元素")
            return False
            
        robot_body_name = robot_body.get('name', robot_name)
        
        # 创建actuator section如果不存在
        actuator_section = root.find('actuator')
        if actuator_section is None:
            actuator_section = ET.SubElement(root, 'actuator')
        
        # 收集所有关节
        joints = []
        for joint in root.iter('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type', 'hinge')
            if joint_name and joint_type in ['hinge', 'slide']:
                joints.append((joint_name, joint_type))
        
        # 为每个关节创建actuator
        for joint_name, joint_type in joints:
            actuator = ET.SubElement(actuator_section, 'general')
            actuator.set('name', f'{joint_name}_actuator')
            actuator.set('joint', joint_name)
            actuator.set('gainprm', '1')
            if joint_type == 'slide':
                actuator.set('ctrlrange', '-1 1')
            else:
                actuator.set('ctrlrange', '-3.14 3.14')
        
        # 创建keyframe section用于初始姿态
        keyframe_section = root.find('keyframe')
        if keyframe_section is None:
            keyframe_section = ET.SubElement(root, 'keyframe')
            
        # 添加default keyframe
        default_key = ET.SubElement(keyframe_section, 'key')
        default_key.set('name', 'home')
        default_key.set('qpos', ' '.join(['0'] * len(joints)))
        
        # 美化XML格式
        def indent(elem, level=0):
            i = "\n" + level*"  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for child in elem:
                    indent(child, level+1)
                if not child.tail or not child.tail.strip():
                    child.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i
        
        indent(root)
        
        # 保存修改后的MJCF
        tree.write(mjcf_path, encoding='utf-8', xml_declaration=True)
        print(f"✅ 已添加 {len(joints)} 个actuator到MJCF文件")
        
        return True
        
    except Exception as e:
        print(f"❌ 添加actor失败: {e}")
        return False


def convert_robot(robot_type, robot_name, project_root):
    """
    为单个机器人执行可靠的转换。

    1. 创建一个独立的临时目录。
    2. 将mesh文件复制到该目录。
    3. 修改URDF，使mesh路径仅为文件名。
    4. cd到临时目录内执行转换。
    5. 将结果复制回主输出目录并清理。
    """
    print(f"--- 转换 {robot_type.upper()}: {robot_name} ---")

    # 1. 定义路径
    if robot_type == 'elfin':
        urdf_path = os.path.join(project_root, 'elfin_description', 'urdf', f'{robot_name}.urdf')
        mesh_source_dir = os.path.join(project_root, 'elfin_description', 'meshes', robot_name)
    elif robot_type == 'ur':
        urdf_path = os.path.join(project_root, 'ur_description', 'urdf', f'{robot_name}.urdf')
        # UR机器人的mesh文件结构比较特殊
        mesh_source_dir = os.path.join(project_root, 'ur_description', 'meshes')
    elif robot_type == 'rohand':
        urdf_path = os.path.join(project_root, 'rohand_urdf_ros2', 'urdf', f'{robot_name}.urdf')
        # rohand根据左右手选择不同的mesh目录
        if 'left' in robot_name:
            mesh_source_dir = os.path.join(project_root, 'rohand_urdf_ros2', 'meshes_l')
        else:  # right hand
            mesh_source_dir = os.path.join(project_root, 'rohand_urdf_ros2', 'meshes_r')
    else:
        print(f"❌ 不支持的机器人类型: {robot_type}")
        return False

    if not os.path.exists(urdf_path):
        print(f"❌ URDF文件不存在: {urdf_path}")
        return False
    if not os.path.exists(mesh_source_dir):
        print(f"❌ Mesh源目录不存在: {mesh_source_dir}")
        return False

    # 2. 创建一个干净的临时转换目录
    temp_dir = os.path.join(project_root, f'./mjcf_models/{robot_name}')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir,exist_ok=True)

    # 3. 复制所有需要的mesh文件到临时目录
    copied_files = 0
    for root, _, files in os.walk(mesh_source_dir):
        for file in files:
            if file.lower().endswith(('.stl', '.dae', '.obj', '.ply')):
                shutil.copy(os.path.join(root, file), temp_dir)
                copied_files += 1
    print(f"📁 复制了 {copied_files} 个mesh文件到临时目录。")

    # 4. 读取URDF并修改mesh路径
    with open(urdf_path, 'r', encoding='utf-8') as f:
        urdf_content = f.read()

    def replace_mesh_path(match):
        # 仅保留文件名
        filename = os.path.basename(match.group(1))
        return f'filename="{filename}"'

    pattern = r'filename="package://[^"]+/([^"]+)"'
    modified_urdf_content = re.sub(pattern, replace_mesh_path, urdf_content)

    temp_urdf_path = os.path.join(temp_dir, f'{robot_name}.urdf')
    with open(temp_urdf_path, 'w', encoding='utf-8') as f:
        f.write(modified_urdf_content)

    # 5. 执行转换
    original_cwd = os.getcwd()
    try:
        # 进入临时目录，这是解决相对路径的关键
        os.chdir(temp_dir)

        # 加载修改后的URDF
        model = mujoco.MjModel.from_xml_path(f'{robot_name}.urdf')

        print("✅ URDF加载成功!")

        # 保存MJCF
        output_mjcf_name = f"{robot_name}.xml"
        mujoco.mj_saveLastXML(output_mjcf_name, model)
        print(f"✅ MJCF文件已生成: {output_mjcf_name}")

        # 6. 添加actor元素到MJCF文件
        if add_actor_to_mjcf(output_mjcf_name, robot_name):
            print(f"✅ 已为 {robot_name} 添加actor元素")
        else:
            print(f"⚠️  添加actor元素失败，但MJCF文件仍然有效")

        # 7. 将结果复制回主输出目录
        # main_output_dir = os.path.join(project_root, "mjcf_models")
        # os.makedirs(main_output_dir, exist_ok=True)
        # final_mjcf_path = os.path.join(main_output_dir, output_mjcf_name)
        # shutil.copy(output_mjcf_name, final_mjcf_path)
        # print(f"📂 MJCF文件已复制到: {os.path.relpath(final_mjcf_path, project_root)}")

        return True

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        # 如果出错，打印临时URDF的内容以供调试
        print("--- 临时URDF内容 (前20行) ---")
        with open(f'{robot_name}.urdf', 'r') as f:
            print("".join(f.readlines()[:20]))
        print("----------------------------")
        return False
    finally:
        # 8. 返回原始目录并清理临时文件
        os.chdir(original_cwd)


def main():
    print("🤖 可靠的URDF转MJCF批量转换工具")
    print("=" * 50)

    project_root = os.getcwd()

    # 定义所有要转换的机器人
    robots_to_convert = []
    # Elfin
    elfin_urdf_dir = "elfin_description/urdf"
    if os.path.isdir(elfin_urdf_dir):
        for f in os.listdir(elfin_urdf_dir):
            if f.endswith('.urdf'):
                robots_to_convert.append(('elfin', f.replace('.urdf', '')))
    # UR
    ur_urdf_dir = "ur_description/urdf"
    if os.path.isdir(ur_urdf_dir):
        for f in os.listdir(ur_urdf_dir):
            if f.endswith('.urdf'):
                robots_to_convert.append(('ur', f.replace('.urdf', '')))
    # RoHand
    rohand_urdf_dir = "rohand_urdf_ros2/urdf"
    if os.path.isdir(rohand_urdf_dir):
        for f in os.listdir(rohand_urdf_dir):
            if f.endswith('.urdf'):
                robots_to_convert.append(('rohand', f.replace('.urdf', '')))

    if not robots_to_convert:
        print("❌ 未找到任何URDF文件，请检查elfin_description、ur_description和rohand_urdf_ros2目录。")
        return

    success_count = 0
    total_count = len(robots_to_convert)
    for robot_type, robot_name in robots_to_convert:
        if convert_robot(robot_type, robot_name, project_root):
            success_count += 1

    print("\n" + "=" * 50)
    print(f"✅ 批量转换完成: {success_count}/{total_count} 成功")

    if success_count > 0:
        print("\n📂 生成的MJCF文件位于 `mjcf_models/` 目录:")
        mjcf_dir = "mjcf_models"
        print("\n💡 使用 `python mjcf_viewer.py mjcf_models/*/<robot_name>.xml` 查看模型。")


if __name__ == '__main__':
    main()