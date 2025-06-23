#!/usr/bin/env python3
"""
双臂机器人逆运动学求解示例脚本

这个脚本展示了如何使用模块化的双臂逆运动学求解包。
可以直接运行此脚本来测试完整的功能。

使用方法:
    python run_example.py

作者: frank
日期: 2024年6月19日
"""

import sys
import os

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dual_arm_ik_modules import main

if __name__ == '__main__':
    print("=" * 60)
    print("双臂机器人逆运动学求解示例")
    print("=" * 60)
    print()
    
    try:
        # 运行主程序
        main()
        
        print("\n" + "=" * 60)
        print("程序执行完成！")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        import traceback
        traceback.print_exc() 