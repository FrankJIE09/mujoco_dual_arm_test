#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人模型组合脚本
将rohand_right机器人手附加到elfin15机器人臂上
"""

import xml.etree.ElementTree as ET
import os
import shutil


def combine_elfin_rohand():
    """
    组合elfin15机器人臂和rohand_right机器人手
    """
    print("🤖 开始组合elfin15机器人臂和rohand_right机器人手...")
    
    # 路径定义
    elfin_path = "mjcf_models/elfin15/elfin15.xml"
    rohand_path = "mjcf_models/rohand_right/rohand_right.xml"
    output_dir = "mjcf_models/elfin15_rohand_right_combined"
    output_path = os.path.join(output_dir, "elfin15_rohand_right.xml")
    
    # 创建输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 解析两个MJCF文件
    print("📖 读取elfin15模型...")
    elfin_tree = ET.parse(elfin_path)
    elfin_root = elfin_tree.getroot()
    
    print("📖 读取rohand_right模型...")
    rohand_tree = ET.parse(rohand_path)
    rohand_root = rohand_tree.getroot()
    
    # 修改模型名称
    elfin_root.set('model', 'elfin15_rohand_combined')
    
    # 1. 合并asset部分
    print("🔗 合并asset部分...")
    elfin_asset = elfin_root.find('asset')
    rohand_asset = rohand_root.find('asset')
    
    # 添加rohand的mesh到elfin的asset中，并给mesh名称加前缀避免冲突
    for mesh in rohand_asset.findall('mesh'):
        mesh_name = mesh.get('name')
        mesh.set('name', f'rohand_{mesh_name}')
        mesh.set('file', f'rohand_{mesh.get("file")}')
        elfin_asset.append(mesh)
    
    # 2. 找到elfin的最后一个link（elfin_link6）
    print("🔍 查找elfin15的末端连接点...")
    elfin_worldbody = elfin_root.find('worldbody')
    
    def find_last_body(body):
        """递归查找最后一个body"""
        children = body.findall('body')
        if not children:
            return body
        return find_last_body(children[-1])
    
    elfin_link1 = elfin_worldbody.find('body[@name="elfin_link1"]')
    last_body = find_last_body(elfin_link1)
    print(f"✅ 找到末端连接点: {last_body.get('name')}")
    
    # 3. 将rohand的body添加到elfin的末端
    print("🤝 附加rohand到elfin末端...")
    rohand_worldbody = rohand_root.find('worldbody')
    
    # 创建一个连接body，调整rohand的位置和姿态
    connection_body = ET.SubElement(last_body, 'body')
    connection_body.set('name', 'hand_attachment')
    connection_body.set('pos', '0.0 0.0 0.171')  # 在末端向上偏移5cm
    connection_body.set('quat', '7.07106781e-01 5.55111512e-17 7.07106781e-01 5.55111512e-17')  # 旋转90度使手掌朝前
    
    # 首先复制rohand的基座几何体
    for geom in rohand_worldbody.findall('geom'):
        new_geom = ET.SubElement(connection_body, 'geom')
        for attr_name, attr_value in geom.attrib.items():
            if attr_name == 'mesh':
                new_geom.set(attr_name, f'rohand_{attr_value}')
            else:
                new_geom.set(attr_name, attr_value)
    
    # 然后复制rohand的所有子body到连接点
    for body in rohand_worldbody.findall('body'):
        # 创建新的body副本
        new_body = ET.SubElement(connection_body, 'body')
        
        # 复制属性
        for attr_name, attr_value in body.attrib.items():
            if attr_name == 'name':
                new_body.set(attr_name, f'rohand_{attr_value}')
            else:
                new_body.set(attr_name, attr_value)
        
        # 递归复制所有子元素
        copy_body_recursive(body, new_body)
    
    # 4. 合并actuator部分
    print("⚙️  合并actuator部分...")
    elfin_actuator = elfin_root.find('actuator')
    rohand_actuator = rohand_root.find('actuator')
    
    # 添加rohand的actuator，修改名称避免冲突
    for motor in rohand_actuator.findall('motor'):
        new_motor = ET.SubElement(elfin_actuator, 'motor')
        motor_name = motor.get('name')
        joint_name = motor.get('joint')
        
        new_motor.set('name', f'rohand_{motor_name}')
        new_motor.set('joint', f'rohand_{joint_name}')
        new_motor.set('gear', motor.get('gear', '1'))
        new_motor.set('ctrlrange', motor.get('ctrlrange', '-3.14 3.14'))
    
    # 5. 更新keyframe
    print("🔑 更新keyframe...")
    elfin_keyframe = elfin_root.find('keyframe')
    key = elfin_keyframe.find('key')
    
    # 扩展qpos以包含rohand的25个关节（elfin有6个，rohand有25个）
    elfin_qpos = "0 0 0 0 0 0"
    rohand_qpos = " ".join(["0"] * 25)
    combined_qpos = f"{elfin_qpos} {rohand_qpos}"
    key.set('qpos', combined_qpos)
    
    # 5.1 为elfin关节添加阻尼参数
    print("⚙️  为elfin关节添加阻尼...")
    def add_damping_to_joints(body):
        """递归为所有关节添加阻尼"""
        for joint in body.findall('joint'):
            # 为旋转关节添加阻尼
            joint.set('damping', '1.0')
            joint.set('frictionloss', '0.1')
        
        # 递归处理子body
        for child_body in body.findall('body'):
            add_damping_to_joints(child_body)
    
    # 为elfin的所有关节添加阻尼
    elfin_link1 = elfin_worldbody.find('body[@name="elfin_link1"]')
    add_damping_to_joints(elfin_link1)
    
    # 6. 复制mesh文件
    print("📁 复制mesh文件...")
    # 复制elfin的mesh文件
    elfin_mesh_dir = "mjcf_models/elfin15"
    for file in os.listdir(elfin_mesh_dir):
        if file.endswith('.STL'):
            shutil.copy(os.path.join(elfin_mesh_dir, file), output_dir)
    
    # 复制rohand的mesh文件并重命名
    rohand_mesh_dir = "mjcf_models/rohand_right"
    for file in os.listdir(rohand_mesh_dir):
        if file.endswith('.STL'):
            new_name = f"rohand_{file}"
            shutil.copy(os.path.join(rohand_mesh_dir, file), 
                       os.path.join(output_dir, new_name))
    
    # 7. 美化XML格式并保存
    print("💾 保存组合模型...")
    indent_xml(elfin_root)
    elfin_tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"✅ 组合完成！文件保存至: {output_path}")
    return output_path


def copy_body_recursive(source_body, target_body):
    """递归复制body及其所有子元素"""
    # 复制所有子元素
    for child in source_body:
        if child.tag == 'body':
            # 递归处理子body
            new_child_body = ET.SubElement(target_body, 'body')
            for attr_name, attr_value in child.attrib.items():
                if attr_name == 'name':
                    new_child_body.set(attr_name, f'rohand_{attr_value}')
                else:
                    new_child_body.set(attr_name, attr_value)
            copy_body_recursive(child, new_child_body)
        elif child.tag == 'joint':
            # 处理关节，修改名称并添加阻尼
            new_joint = ET.SubElement(target_body, 'joint')
            for attr_name, attr_value in child.attrib.items():
                if attr_name == 'name':
                    new_joint.set(attr_name, f'rohand_{attr_value}')
                else:
                    new_joint.set(attr_name, attr_value)
            
            # 添加阻尼参数以保持关节稳定
            if child.get('type') == 'slide':
                # 滑动关节使用较小的阻尼
                new_joint.set('damping', '0.1')
                new_joint.set('frictionloss', '0.01')
            else:
                # 旋转关节使用适中的阻尼
                new_joint.set('damping', '0.5')
                new_joint.set('frictionloss', '0.05')
        elif child.tag == 'geom':
            # 处理几何体，修改mesh名称
            new_geom = ET.SubElement(target_body, 'geom')
            for attr_name, attr_value in child.attrib.items():
                if attr_name == 'mesh':
                    new_geom.set(attr_name, f'rohand_{attr_value}')
                else:
                    new_geom.set(attr_name, attr_value)
        else:
            # 其他元素直接复制
            new_child = ET.SubElement(target_body, child.tag)
            new_child.text = child.text
            new_child.tail = child.tail
            for attr_name, attr_value in child.attrib.items():
                new_child.set(attr_name, attr_value)


def indent_xml(elem, level=0):
    """美化XML格式"""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def main():
    """主函数"""
    try:
        combined_model_path = combine_elfin_rohand()
        
        print("\n🎉 模型组合成功！")
        print(f"📂 组合模型位于: {combined_model_path}")
        print("\n🚀 使用以下命令查看组合模型:")
        print(f"python mjcf_viewer.py {combined_model_path}")
        
        # 询问是否立即启动viewer
        response = 'y'
        if response in ['y', 'yes']:
            os.system(f"python mjcf_viewer.py {combined_model_path}")
            
    except Exception as e:
        print(f"❌ 组合失败: {e}")


if __name__ == "__main__":
    main() 