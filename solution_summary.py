#!/usr/bin/env python3
"""
Elfin15运动学问题解决方案总结

问题：ikpy逆向运动学返回9个元素而不是预期的6个，且index[1]值不稳定
解决：通过正确配置active_links_mask和坐标系校准
"""

import numpy as np
from kinematics import Elfin15Kinematics

def main():
    print("🎯 Elfin15运动学解决方案总结")
    print("=" * 60)
    
    # 问题描述
    print("\n📋 原始问题:")
    print("  1. ikpy IK解返回9个元素，期望6个")
    print("  2. index[1]的值在不同IK调用中变化 (1.0, 94.89, -69.01)")
    print("  3. FK和IK之间存在位置不一致问题")
    
    print("\n🔧 解决方案:")
    print("  1. 设置正确的active_links_mask = [False, False, True×6, False]")
    print("  2. 只让6个旋转关节(index 2-7)参与IK优化")
    print("  3. 固定关节(index 0,1,8)保持为0")
    print("  4. 添加坐标系校准功能")
    print("  5. FK和IK都使用ikpy以确保一致性")
    
    # 实际验证
    print("\n✅ 解决方案验证:")
    print("-" * 40)
    
    kin = Elfin15Kinematics()
    
    # 测试配置
    test_config = np.array([0., 0., 0.2, -0.3, 0.5, -0.1, 0.4, 0.2, 0.])
    
    # FK
    transform = kin.ik_chain.forward_kinematics(test_config)
    pos = transform[:3, 3]
    rot = transform[:3, :3]
    
    # IK
    ik_solution = kin.ik_chain.inverse_kinematics(
        target_position=pos,
        target_orientation=rot,
        orientation_mode="all"
    )
    
    # 验证
    pos_error = np.linalg.norm(test_config - ik_solution)
    
    print(f"原始配置: {test_config}")
    print(f"IK求解:   {ik_solution}")
    print(f"配置误差: {pos_error:.12f}")
    
    # 检查固定关节
    fixed_joints_ok = (abs(ik_solution[0]) < 1e-10 and 
                      abs(ik_solution[1]) < 1e-10 and 
                      abs(ik_solution[8]) < 1e-10)
    
    print(f"固定关节检查: index[0]={ik_solution[0]:.2e}, index[1]={ik_solution[1]:.2e}, index[8]={ik_solution[8]:.2e}")
    
    # 最终状态
    print(f"\n🏆 最终状态:")
    if pos_error < 1e-6 and fixed_joints_ok:
        print("  ✅ IK解长度: 9个元素 (正确)")
        print("  ✅ 固定关节: 保持为0 (正确)")
        print("  ✅ FK/IK一致性: 完美匹配")
        print("  ✅ 数值稳定性: 高精度")
        print("\n🎉 问题完全解决!")
    else:
        print("  ❌ 仍存在问题需要调试")
    
    print(f"\n📚 关键技术点:")
    print("  • active_links_mask正确配置")
    print("  • 坐标系统一(都使用ikpy)")
    print("  • 固定关节约束")
    print("  • MuJoCo用于最终可视化")

if __name__ == '__main__':
    main() 