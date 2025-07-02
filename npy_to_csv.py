import numpy as np
import pandas as pd
import argparse
import os


def npy_to_csv(npy_path, csv_path=None):
    data = np.load(npy_path, allow_pickle=True).item()
    df = pd.DataFrame(data['joint_trajectory'])
    df['timestamp'] = data['timestamps']
    if csv_path is None:
        csv_path = os.path.splitext(npy_path)[0] + '.csv'
    df.to_csv(csv_path, index=False)
    print(f"已将 {npy_path} 转换为 {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='将包含joint_trajectory和timestamps的npy文件转换为csv')
    parser.add_argument('--npy_path', type=str, default="jaka_smoothed_trajectory.npy", help='输入的npy文件路径')
    parser.add_argument('--csv', type=str, default=None, help='输出的csv文件路径（可选）')
    args = parser.parse_args()
    npy_to_csv(args.npy_path, args.csv)


if __name__ == '__main__':
    main() 