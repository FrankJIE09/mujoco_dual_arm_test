#!/usr/bin/env python3
"""
matplotlib中文字体配置模块

该模块提供了matplotlib中文字体的检测、配置和管理功能，支持：
1. 自动检测系统中可用的中文字体文件
2. 设置自定义字体文件路径
3. 跨平台字体支持 (Linux, Windows, macOS)

作者: frank
日期: 2024年6月19日
"""

import os
import platform
import matplotlib.pyplot as plt
from matplotlib import font_manager


def list_available_chinese_fonts():
    """
    列出系统中所有可用的中文字体文件
    
    Returns:
        list: 中文字体文件路径列表
    """
    system = platform.system()
    
    if system == "Linux":
        possible_paths = [
            "/usr/share/fonts/",
            "/usr/local/share/fonts/", 
            "~/.fonts/",
            "/usr/share/fonts/truetype/",
            "/usr/share/fonts/opentype/"
        ]
    elif system == "Windows":
        possible_paths = [
            "C:/Windows/Fonts/",
            "C:/Windows/System32/Fonts/"
        ]
    elif system == "Darwin":  # macOS
        possible_paths = [
            "/System/Library/Fonts/",
            "/Library/Fonts/",
            "~/Library/Fonts/"
        ]
    else:
        possible_paths = []
    
    chinese_fonts = []
    for path in possible_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            for root, dirs, files in os.walk(expanded_path):
                for file in files:
                    if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                        # 检查是否为中文字体
                        if any(keyword in file.lower() for keyword in [
                            'simhei', 'simsun', 'microsoftyahei', 'yahei', 'heiti', 
                            'songti', 'kaiti', 'fangsong', 'wenquanyi', 'noto', 
                            'droid', 'source', 'han', 'cjk', 'arphic', 'uming', 
                            'ukai', 'wqy', 'zenhei'
                        ]):
                            font_path = os.path.join(root, file)
                            chinese_fonts.append(font_path)
    
    return chinese_fonts


def get_preferred_fonts():
    """
    获取推荐的中文字体列表（按优先级排序）
    
    Returns:
        list: 推荐字体文件路径列表
    """
    system = platform.system()
    
    if system == "Linux":
        preferred_fonts = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',           # 文泉驿正黑
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', # Noto Sans CJK
            '/usr/share/fonts/truetype/arphic/uming.ttc',             # AR PL UMing
            '/usr/share/fonts/truetype/arphic/ukai.ttc',              # AR PL UKai
            '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc' # Noto Serif CJK
        ]
    elif system == "Windows":
        preferred_fonts = [
            'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/simkai.ttf'   # 楷体
        ]
    elif system == "Darwin":  # macOS
        preferred_fonts = [
            '/System/Library/Fonts/PingFang.ttc',           # 苹方
            '/System/Library/Fonts/STHeiti Light.ttc',      # 华文黑体
            '/Library/Fonts/Arial Unicode MS.ttf',         # Arial Unicode MS
            '/System/Library/Fonts/Hiragino Sans GB.ttc'   # 冬青黑体
        ]
    else:
        preferred_fonts = []
    
    return preferred_fonts


def set_custom_font(font_path):
    """
    设置自定义字体文件
    
    Args:
        font_path (str): 字体文件路径 (.ttf, .ttc, .otf)
        
    Returns:
        bool: 设置成功返回True，失败返回False
    """
    if not os.path.exists(font_path):
        print(f"❌ 字体文件不存在: {font_path}")
        return False
        
    try:
        # 注册字体
        font_manager.fontManager.addfont(font_path)
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        
        print(f"✅ 成功设置自定义字体: {font_prop.get_name()}")
        print(f"   字体文件: {font_path}")
        return True
        
    except Exception as e:
        print(f"❌ 设置自定义字体失败: {e}")
        return False


def setup_chinese_font(verbose=True):
    """
    自动设置matplotlib中文字体支持
    
    Args:
        verbose (bool): 是否显示详细信息
        
    Returns:
        bool: 设置成功返回True，失败返回False
    """
    if verbose:
        print("🔍 正在检测系统中文字体...")
    
    # 首先尝试推荐字体
    preferred_fonts = get_preferred_fonts()
    
    for font_path in preferred_fonts:
        if os.path.exists(font_path):
            if verbose:
                print(f"  ✅ 找到推荐字体: {os.path.basename(font_path)}")
            try:
                font_manager.fontManager.addfont(font_path)
                font_prop = font_manager.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                if verbose:
                    print(f"  🎯 已设置字体: {font_prop.get_name()}")
                    print(f"     文件路径: {font_path}")
                return True
            except Exception as e:
                if verbose:
                    print(f"  ⚠️  设置字体失败: {e}")
                continue
    
    # 如果推荐字体都不可用，搜索所有可用字体
    if verbose:
        print("  🔄 推荐字体不可用，搜索其他中文字体...")
    
    all_fonts = list_available_chinese_fonts()
    
    if verbose and all_fonts:
        print(f"  📋 检测到 {len(all_fonts)} 个中文字体文件:")
        for i, font_path in enumerate(all_fonts[:5]):  # 只显示前5个
            print(f"     {i+1}. {os.path.basename(font_path)}")
        if len(all_fonts) > 5:
            print(f"     ... 还有 {len(all_fonts)-5} 个字体")
    
    # 尝试使用第一个找到的字体
    for font_path in all_fonts:
        try:
            font_manager.fontManager.addfont(font_path)
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            if verbose:
                print(f"  ✅ 已设置字体: {font_prop.get_name()}")
                print(f"     文件路径: {font_path}")
            return True
        except Exception as e:
            continue
    
    # 如果没有找到字体文件，使用系统字体名称作为备用方案
    if verbose:
        print("  ⚠️  未找到可用的字体文件，使用备用字体设置...")
    
    system = platform.system()
    if system == "Linux":
        backup_fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
    elif system == "Windows":
        backup_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
    elif system == "Darwin":
        backup_fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
    else:
        backup_fonts = ['DejaVu Sans']
    
    plt.rcParams['font.sans-serif'] = backup_fonts
    
    if verbose:
        print(f"  🔧 已设置备用字体: {backup_fonts}")
    
    return False


def test_chinese_display():
    """
    测试中文字体显示效果
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("🧪 测试中文字体显示效果...")
        
        # 创建简单的测试图
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 测试文本
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        ax.plot(x, y, 'b-', linewidth=2, label='正弦曲线')
        ax.set_xlabel('横坐标 (X轴)')
        ax.set_ylabel('纵坐标 (Y轴)')
        ax.set_title('中文字体显示测试 - 数学函数图像')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加中文注释
        ax.text(5, 0.5, '这是中文测试文本\n包含数字123和符号！@#', 
                fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('font_test.png', dpi=150, bbox_inches='tight')
        print("✅ 中文字体测试完成，已保存为 font_test.png")
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ 中文字体测试失败: {e}")
        return False


def init_chinese_font(font_path=None, verbose=True):
    """
    初始化中文字体支持（推荐的入口函数）
    
    Args:
        font_path (str, optional): 指定字体文件路径，如果为None则自动检测
        verbose (bool): 是否显示详细信息
        
    Returns:
        bool: 初始化成功返回True，失败返回False
    """
    # 设置负号显示
    plt.rcParams['axes.unicode_minus'] = False
    
    if font_path:
        # 使用指定的字体文件
        result = set_custom_font(font_path)
    else:
        # 自动检测并设置字体
        result = setup_chinese_font(verbose)
    
    if verbose:
        if result:
            print("🎉 中文字体初始化成功！")
        else:
            print("⚠️  中文字体初始化完成（使用备用方案）")
    
    return result


def print_font_info():
    """
    打印当前字体设置信息
    """
    print("📊 当前matplotlib字体设置:")
    print(f"   font.family: {plt.rcParams['font.family']}")
    print(f"   font.sans-serif: {plt.rcParams['font.sans-serif']}")
    print(f"   axes.unicode_minus: {plt.rcParams['axes.unicode_minus']}")


if __name__ == '__main__':
    """
    如果直接运行此文件，展示字体配置功能
    """
    print("=" * 60)
    print("matplotlib中文字体配置工具")
    print("=" * 60)
    
    # 列出可用字体
    fonts = list_available_chinese_fonts()
    print(f"\n🔍 系统中检测到 {len(fonts)} 个中文字体文件:")
    for i, font_path in enumerate(fonts):
        print(f"  {i+1:2d}. {os.path.basename(font_path):30s} -> {font_path}")
    
    print(f"\n🎯 推荐字体列表:")
    preferred = get_preferred_fonts()
    for i, font_path in enumerate(preferred):
        exists = "✅" if os.path.exists(font_path) else "❌"
        print(f"  {i+1}. {exists} {os.path.basename(font_path)}")
    
    # 初始化字体
    print(f"\n🚀 初始化中文字体支持:")
    init_chinese_font()
    
    # 显示当前设置
    print(f"\n")
    print_font_info()
    
    # 测试字体显示
    print(f"\n")
    test_chinese_display() 