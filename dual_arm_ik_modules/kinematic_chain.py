"""
运动学链构建模块

这个模块负责从MuJoCo XML文件解析机器人模型并创建ikpy运动学链。
主要功能包括：
1. 解析MuJoCo XML文件结构
2. 提取关节和连杆信息
3. 创建ikpy运动学链对象
4. 处理双臂机器人的运动学模型

作者: frank
日期: 2024年6月19日
"""

import ikpy.chain
import ikpy.link
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from .transform_utils import get_transformation


def create_chain_from_mjcf(xml_file, base_body_name):
    """
    解析MuJoCo XML文件，为指定的机械臂创建ikpy运动学链。
    
    这个函数遍历XML中的连杆结构，提取关节信息，并创建ikpy的Chain对象。
    支持6自由度机械臂，每个关节都有位置和姿态限制。
    
    运动学链结构：
    base → link1 → link2 → ... → link6 → end_effector
    每个连杆的变换: T_i = [R_i    p_i]
                          [0      1  ]
    
    Args:
        xml_file (str): MuJoCo XML文件的路径
        base_body_name (str): 机械臂基座的body名称 (例如, 'left_robot_base')
    
    Returns:
        tuple: (ikpy.chain.Chain, np.ndarray) 
               - 创建的运动学链
               - 基座在世界坐标系下的变换矩阵
    """
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()
    worldbody = root.find('worldbody')

    # 找到机械臂的基座元素并获取其在世界坐标系下的变换
    base_element = worldbody.find(f".//body[@name='{base_body_name}']")
    base_transform = get_transformation(base_element)

    # 创建ikpy链的连杆列表
    # 第一个连杆通常是固定的世界坐标系或基座
    links = [ikpy.link.URDFLink(
        name="base",
        origin_translation=[0, 0, 0],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0]
    )]

    # active_links_mask 用于告诉ikpy哪些关节是可动的
    # 第一个基座连杆是固定的 (False)
    active_links_mask = [False]

    # 从基座开始，迭代遍历6个连杆
    current_element = base_element
    for i in range(1, 7):
        # 构建连杆名称 (例如: left_link1, left_link2, ...)
        link_name = f"{base_body_name.split('_')[0]}_link{i}"
        current_element = current_element.find(f".//body[@name='{link_name}']")
        if current_element is None:
            break

        # 从XML中提取关节信息
        joint_element = current_element.find('joint')
        joint_name = joint_element.get('name')
        joint_axis = np.fromstring(joint_element.get('axis'), sep=' ')
        joint_range = joint_element.get('range')
        bounds = tuple(map(float, joint_range.split())) if joint_range else (None, None)

        # 获取连杆相对于其父连杆的变换
        link_transform = get_transformation(current_element)
        translation = link_transform[:3, 3]  # 平移部分
        orientation_matrix = link_transform[:3, :3]  # 旋转部分
        # 将旋转矩阵转换为ikpy期望的RPY欧拉角格式
        # 数学公式: [φ, θ, ψ] = matrix_to_euler(R, 'xyz')
        orientation_rpy = Rotation.from_matrix(orientation_matrix).as_euler('xyz')

        # 创建ikpy连杆对象
        link = ikpy.link.URDFLink(
            name=joint_name,
            origin_translation=translation,  # 连杆原点相对于父连杆的平移
            origin_orientation=orientation_rpy,  # 连杆原点相对于父连杆的旋转(RPY)
            rotation=joint_axis,  # 关节的旋转轴
            bounds=bounds  # 关节角度限制
        )
        links.append(link)
        active_links_mask.append(True)  # 这6个关节都是可动的

    # 添加末端执行器标记 (如果有的话)，它是一个固定连杆
    ee_body = None
    for body in current_element.iter('body'):
        if '_end_effector' in body.get('name', ''):
            ee_body = body
            break

    if ee_body is not None:
        ee_transform = get_transformation(ee_body)
        ee_orientation_matrix = ee_transform[:3, :3]
        ee_orientation_rpy = Rotation.from_matrix(ee_orientation_matrix).as_euler('xyz')
        ee_link = ikpy.link.URDFLink(
            name=ee_body.get('name'),
            origin_translation=ee_transform[:3, 3],
            origin_orientation=ee_orientation_rpy,
            rotation=[0, 0, 0]  # 固定连杆，无旋转
        )
        links.append(ee_link)
        active_links_mask.append(False)  # 末端执行器标记是固定的

    # 使用创建的连杆列表和活动关节掩码来构建运动学链
    chain = ikpy.chain.Chain(links, active_links_mask=active_links_mask)
    return chain, base_transform


def get_kinematics(xml_file_path):
    """
    为双臂机器人创建运动学模型。
    
    这个函数为左右两个机械臂分别创建运动学链，并返回它们的基座变换矩阵。
    
    Args:
        xml_file_path (str): MuJoCo XML文件的路径
    
    Returns:
        tuple: (左臂链, 右臂链, 左臂基座变换, 右臂基座变换)
    """
    left_chain, left_base_transform = create_chain_from_mjcf(xml_file_path, 'left_robot_base')
    right_chain, right_base_transform = create_chain_from_mjcf(xml_file_path, 'right_robot_base')
    return left_chain, right_chain, left_base_transform, right_base_transform 