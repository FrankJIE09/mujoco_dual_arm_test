"""
双臂机器人逆运动学求解模块包

这个包提供了完整的双臂机器人闭环逆运动学求解功能。
包含以下模块：
- transform_utils: 坐标变换工具函数
- kinematic_chain: 运动学链构建
- ik_solver: 逆运动学求解算法
- visualization: 可视化和结果展示
- main: 主程序入口

作者: frank
日期: 2024年6月19日
"""

from .transform_utils import (
    rpy_to_matrix,
    quat_to_matrix,
    pose_to_transformation_matrix,
    transformation_matrix_to_pose,
    get_transformation
)

from .kinematic_chain import (
    create_chain_from_mjcf,
    get_kinematics
)

from .ik_solver import (
    solve_dual_arm_ik
)

from .visualization import (
    plot_iteration_history,
    show_mujoco_iteration_process,
    show_mujoco_home_pose,
    print_optimization_statistics
)

from .main import main

__version__ = "1.0.0"
__author__ = "frank"

__all__ = [
    # 变换工具函数
    'rpy_to_matrix',
    'quat_to_matrix', 
    'pose_to_transformation_matrix',
    'transformation_matrix_to_pose',
    'get_transformation',
    
    # 运动学链
    'create_chain_from_mjcf',
    'get_kinematics',
    
    # 逆运动学求解
    'solve_dual_arm_ik',
    
    # 可视化
    'plot_iteration_history',
    'show_mujoco_iteration_process',
    'show_mujoco_home_pose',
    'print_optimization_statistics',
    
    # 主程序
    'main'
] 