import numpy as np
import pandas as pd
import argparse
import os
import glob


def npy_to_csv(npy_path, csv_path=None):
    """将单个npy文件转换为csv文件"""
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        
        # 检查数据格式
        if 'joint_trajectory' in data and 'timestamps' in data:
            df = pd.DataFrame(data['joint_trajectory'])
            df['timestamp'] = data['timestamps']
        elif isinstance(data, np.ndarray):
            # 如果直接是数组，假设是关节轨迹
            df = pd.DataFrame(data)
            df['timestamp'] = np.arange(len(data)) * 0.02  # 假设20ms时间步长
        else:
            print(f"警告：{npy_path} 数据格式不支持，跳过")
            return False
            
        if csv_path is None:
            csv_path = os.path.splitext(npy_path)[0] + '.csv'
        
        df.to_csv(csv_path, index=False)
        print(f"✅ 已将 {npy_path} 转换为 {csv_path}")
        return True
        
    except Exception as e:
        print(f"❌ 转换 {npy_path} 失败: {e}")
        return False


def convert_all_npy_files():
    """转换当前文件夹中所有的npy文件"""
    # 获取当前文件夹中的所有npy文件（不包含子文件夹）
    npy_files = glob.glob("*.npy")
    
    if not npy_files:
        print("当前文件夹中没有找到npy文件")
        return
    
    print(f"找到 {len(npy_files)} 个npy文件:")
    for file in npy_files:
        print(f"  - {file}")
    
    print("\n开始转换...")
    success_count = 0
    
    for npy_file in npy_files:
        if npy_to_csv(npy_file):
            success_count += 1
    
    print(f"\n转换完成！成功转换 {success_count}/{len(npy_files)} 个文件")


def main():
    parser = argparse.ArgumentParser(description='将npy文件转换为csv文件')
    parser.add_argument('--npy_path', type=str, help='单个npy文件路径（可选）')
    parser.add_argument('--csv', type=str, help='输出的csv文件路径（可选）')
    parser.add_argument('--all', action='store_true', help='转换当前文件夹中所有的npy文件')
    args = parser.parse_args()
    
    if args.all:
        # 转换所有npy文件
        convert_all_npy_files()
    elif args.npy_path:
        # 转换单个文件
        npy_to_csv(args.npy_path, args.csv)
    else:
        # 默认转换所有npy文件
        convert_all_npy_files()


if __name__ == '__main__':
    main() 