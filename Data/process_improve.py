import os


def improved_preprocessing(num_turbines=50):
    """
    改进的数据预处理
    """
    print("\n" + "=" * 60)
    print("Step 2: 改进的数据预处理")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from tqdm import tqdm
    import pickle

    # 1. 加载清洗后的数据
    print("\n📁 加载清洗后的数据...")
    df = pd.read_csv('sdwpf_cleaned.csv')

    # 2. 选择涡轮机
    print(f"\n🎯 选择前 {num_turbines} 个涡轮机...")
    selected_turbines = sorted(df['TurbID'].unique())[:num_turbines]
    df_selected = df[df['TurbID'].isin(selected_turbines)]
    print(f"选中的涡轮机: {selected_turbines[:5]}...{selected_turbines[-5:]}")

    # 3. 特征列表
    feature_columns = [
        'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir',
        'Pab1', 'Pab2', 'Pab3', 'Prtv', 'T2m',
        'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp', 'Patv'
    ]

    # 4. 归一化策略
    print("\n🔧 应用RobustScaler归一化...")
    scalers = {}
    processed_data = {}

    for tid in tqdm(selected_turbines, desc="处理涡轮机"):
        turbine_df = df_selected[df_selected['TurbID'] == tid]

        # 使用RobustScaler（对异常值更鲁棒）
        scaler = RobustScaler()
        normalized_data = scaler.fit_transform(turbine_df[feature_columns])

        processed_data[tid] = normalized_data
        scalers[f'turbine_{tid}'] = scaler

    # 5. 创建空间网格（优化大小）
    print("\n📐 创建优化空间网格...")

    # 计算合适的网格大小
    if num_turbines <= 20:
        grid_shape = (4, 5)
    elif num_turbines <= 50:
        grid_shape = (7, 8)
    else:
        grid_shape = (10, 14)

    H, W = grid_shape
    print(f"网格大小: {H}×{W} (容纳{H * W}个位置)")

    # 6. 创建空间帧
    print("\n🎨 创建空间帧...")

    # 获取时间步数
    T = next(iter(processed_data.values())).shape[0]
    C = len(feature_columns)

    frames = np.zeros((T, H, W, C), dtype=np.float32)

    # 映射涡轮机到网格
    positions = {}
    for idx, tid in enumerate(selected_turbines):
        row = idx // W
        col = idx % W
        positions[tid] = (row, col)
        frames[:, row, col, :] = processed_data[tid]

    print(f"空间帧形状: {frames.shape}")
    print(f"活跃位置: {len(positions)}/{H * W}")

    # 7. 数据集划分（60/20/20）
    print("\n📊 划分数据集...")

    n_samples = len(frames)
    train_size = int(n_samples * 0.6)
    val_size = int(n_samples * 0.2)

    train_frames = frames[:train_size]
    val_frames = frames[train_size:train_size + val_size]
    test_frames = frames[train_size + val_size:]

    print(f"训练集: {train_frames.shape}")
    print(f"验证集: {val_frames.shape}")
    print(f"测试集: {test_frames.shape}")

    # 8. 保存
    print("\n💾 保存预处理数据...")

    np.savez_compressed(
        'improved_preprocessed_data.npz',
        train_frames=train_frames,
        val_frames=val_frames,
        test_frames=test_frames,
        grid_shape=grid_shape,
        positions=positions,
        feature_columns=feature_columns
    )

    with open('improved_scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

    print("✅ 预处理完成!")

    return train_frames, val_frames, test_frames

if __name__ == '__main__':
    if not os.path.exists('improved_preprocessed_data.npz'):
        improved_preprocessing(num_turbines=50)
    else:
        print("✅ 预处理数据已存在，跳过Step 2")