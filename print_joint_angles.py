import numpy as np

# 轨迹文件名（可根据实际情况修改）
filename = 'jaka_interpolated_trajectory.npy'

def main():
    # 加载npy文件
    data = np.load(filename, allow_pickle=True).item()
    joint_trajectory = data['joint_trajectory']
    timestamps = data.get('timestamps', None)

    print(f"轨迹点数: {len(joint_trajectory)}")
    print("每一行为一个时间点的12个关节角度（单位：弧度）：\n")
    for i, q in enumerate(joint_trajectory):
        if timestamps is not None:
            print(f"t={timestamps[i]:.3f}s: ", end='')
        print(' '.join(f"{angle:.6f}" for angle in q))

if __name__ == '__main__':
    main() 