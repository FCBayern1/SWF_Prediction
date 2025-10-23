import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class SDWPFPreprocessor:
    """SDWPF数据预处理类"""

    def __init__(self):
        self.scalers = {}
        self.feature_columns = None

    def load_data(self, filepath):
        """加载数据"""
        print("Loading SDWPF data...")

        # 指定数据类型
        dtypes = {
            'TurbID': np.int32,
            'Wspd': np.float32,
            'Wdir': np.float32,
            'Etmp': np.float32,
            'Itmp': np.float32,
            'Ndir': np.float32,
            'Pab1': np.float32,
            'Pab2': np.float32,
            'Pab3': np.float32,
            'Prtv': np.float32,
            'T2m': np.float32,
            'Sp': np.float32,
            'RelH': np.float32,
            'Wspd_w': np.float32,
            'Wdir_w': np.float32,
            'Tp': np.float32,
            'Patv': np.float32
        }

        # 读取数据
        df = pd.read_csv(filepath, parse_dates=['Tmstamp'], dtype=dtypes)
        df = df.sort_values(['TurbID', 'Tmstamp'])

        print(f"  Shape: {df.shape}")
        print(f"  Turbines: {df['TurbID'].nunique()}")
        print(f"  Time range: {df['Tmstamp'].min()} to {df['Tmstamp'].max()}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        return df

    def clean_data(self, df):
        """数据清洗"""
        print("\nCleaning data...")

        # 处理负功率
        neg_mask = df['Patv'] < 0
        neg_count = neg_mask.sum()
        df.loc[neg_mask, 'Patv'] = 0
        print(f"  Fixed {neg_count} negative power values")

        # 外部温度
        etmp_mask = (df['Etmp'] >= -50) & (df['Etmp'] <= 50)
        etmp_median = df.loc[etmp_mask, 'Etmp'].median()
        etmp_outliers = (~etmp_mask).sum()
        df.loc[~etmp_mask, 'Etmp'] = etmp_median
        print(f"  Fixed {etmp_outliers} Etmp outliers")

        # 内部温度
        itmp_mask = (df['Itmp'] >= -50) & (df['Itmp'] <= 80)
        itmp_median = df.loc[itmp_mask, 'Itmp'].median()
        itmp_outliers = (~itmp_mask).sum()
        df.loc[~itmp_mask, 'Itmp'] = itmp_median
        print(f"  Fixed {itmp_outliers} Itmp outliers")

        return df

    def select_features(self, df):
        """特征选择"""
        self.feature_columns = [
            'Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir',
            'Pab1', 'Pab2', 'Pab3', 'Prtv', 'T2m',
            'Sp', 'RelH', 'Wspd_w', 'Wdir_w', 'Tp',
            'Patv'
        ]

        print(f"\nSelected {len(self.feature_columns)} features")

        columns_to_keep = ['TurbID', 'Tmstamp'] + self.feature_columns
        df = df[columns_to_keep]

        return df

    def interpolate_missing_fast(self, df):
        """快速插值处理"""
        print("\nInterpolating missing values...")

        def interpolate_group(group):
            for col in self.feature_columns:
                group[col] = group[col].interpolate(method='linear', limit_direction='both')
            return group

        tqdm.pandas(desc="Interpolating")
        df = df.groupby('TurbID', group_keys=False).progress_apply(interpolate_group)

        # 填充剩余NaN
        df[self.feature_columns] = df[self.feature_columns].fillna(method='bfill').fillna(method='ffill')

        return df

    def normalize_data_fast(self, df, train_times):
        """快速归一化"""
        print("\nNormalizing data...")

        normalized_data = {}
        range_minus_one = ['Wdir', 'Etmp', 'Itmp', 'T2m']

        # 预计算全局训练统计
        global_train = df[df['Tmstamp'].isin(train_times)]
        global_scalers = {}

        for feature in self.feature_columns:
            if feature in range_minus_one:
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))

            scaler.fit(global_train[feature].values.reshape(-1, 1))
            global_scalers[feature] = scaler

        # 处理每个涡轮机
        turbine_ids = df['TurbID'].unique()

        for turbine_id in tqdm(turbine_ids, desc="Normalizing turbines"):
            turbine_df = df[df['TurbID'] == turbine_id].copy()
            turbine_train = turbine_df[turbine_df['Tmstamp'].isin(train_times)]

            use_global = len(turbine_train) < 100

            normalized_features = []

            for feature in self.feature_columns:
                if use_global:
                    scaler = global_scalers[feature]
                else:
                    if feature in range_minus_one:
                        scaler = MinMaxScaler(feature_range=(-1, 1))
                    else:
                        scaler = MinMaxScaler(feature_range=(0, 1))

                    train_vals = turbine_train[feature].values.reshape(-1, 1)
                    scaler.fit(train_vals)

                all_vals = turbine_df[feature].values.reshape(-1, 1)
                normalized = scaler.transform(all_vals)
                normalized_features.append(normalized)

                self.scalers[f'turbine_{turbine_id}_{feature}'] = scaler

            normalized_array = np.hstack(normalized_features)

            normalized_data[turbine_id] = {
                'data': normalized_array.astype(np.float32),
                'timestamps': turbine_df['Tmstamp'].values  # 保持pandas时间戳格式
            }

        return normalized_data

    def create_spatial_frames_fast(self, normalized_data, H=28, W=6):
        """快速创建时空帧"""
        print(f"\nCreating spatiotemporal frames ({H}x{W})...")

        # 创建涡轮机位置映射
        turbine_ids = sorted(normalized_data.keys())
        turbine_positions = {}

        for idx, tid in enumerate(turbine_ids):
            h = idx // W
            w = idx % W
            if h < H:
                turbine_positions[tid] = (h, w)

        print(f"  Placed {len(turbine_positions)} turbines in grid")

        # 获取所有唯一时间戳
        all_timestamps = set()
        for tid in normalized_data:
            all_timestamps.update(normalized_data[tid]['timestamps'])
        all_timestamps = sorted(all_timestamps)
        print(f"  Total timestamps: {len(all_timestamps)}")

        # 创建时间戳索引映射
        print("  Building timestamp indices...")
        timestamp_to_idx = {}
        for tid in turbine_positions.keys():
            timestamp_to_idx[tid] = {
                ts: idx for idx, ts in enumerate(normalized_data[tid]['timestamps'])
            }

        # 预分配数组
        n_features = len(self.feature_columns)
        frames = np.zeros((len(all_timestamps), H, W, n_features), dtype=np.float32)

        # 填充数据
        print("  Filling frames...")
        for t_idx, timestamp in enumerate(tqdm(all_timestamps, desc="Creating frames")):
            for tid, (h, w) in turbine_positions.items():
                if tid in timestamp_to_idx and timestamp in timestamp_to_idx[tid]:
                    data_idx = timestamp_to_idx[tid][timestamp]
                    frames[t_idx, h, w, :] = normalized_data[tid]['data'][data_idx]

        return frames, np.array(all_timestamps), turbine_positions

    def process(self, filepath, output_prefix='sdwpf_paper',
                train_ratio=0.8, val_ratio=0.1):
        """完整的数据处理流程"""

        print("=" * 60)
        print("SDWPF Data Preprocessing (Fixed)")
        print("=" * 60)

        # 1. 加载数据
        df = self.load_data(filepath)

        # 2. 清洗数据
        df = self.clean_data(df)

        # 3. 特征选择
        df = self.select_features(df)

        # 4. 插值处理
        df = self.interpolate_missing_fast(df)

        # 5. 数据分割 - 使用时间戳
        print("\nSplitting data by time...")
        unique_times = sorted(df['Tmstamp'].unique())
        n_times = len(unique_times)

        train_end = int(n_times * train_ratio)
        val_end = int(n_times * (train_ratio + val_ratio))

        train_times = unique_times[:train_end]
        val_times = unique_times[train_end:val_end]
        test_times = unique_times[val_end:]

        print(f"  Total unique timestamps: {n_times}")
        print(f"  Train: {len(train_times)} timestamps ({train_ratio * 100:.0f}%)")
        print(f"  Val: {len(val_times)} timestamps ({val_ratio * 100:.0f}%)")
        print(f"  Test: {len(test_times)} timestamps ({(1 - train_ratio - val_ratio) * 100:.0f}%)")

        # 验证分割
        print(f"  Train time range: {train_times[0]} to {train_times[-1]}")
        print(f"  Val time range: {val_times[0]} to {val_times[-1]}")
        print(f"  Test time range: {test_times[0]} to {test_times[-1]}")

        # 6. 归一化
        normalized_data = self.normalize_data_fast(df, train_times)

        # 7. 创建时空帧
        frames, timestamps, turbine_positions = self.create_spatial_frames_fast(normalized_data)

        # 8. 分割数据集 - 修复时间戳类型不匹配问题
        print("\nSplitting frames...")

        # 转换时间戳类型以确保一致性
        # 将numpy datetime64转换为pandas Timestamp进行比较
        timestamps_pd = pd.to_datetime(timestamps)

        # 创建时间戳集合以提高查找效率
        train_times_set = set(train_times)
        val_times_set = set(val_times)
        test_times_set = set(test_times)

        # 使用集合查找创建mask
        train_mask = np.array([ts in train_times_set for ts in timestamps_pd])
        val_mask = np.array([ts in val_times_set for ts in timestamps_pd])
        test_mask = np.array([ts in test_times_set for ts in timestamps_pd])

        print(f"  Train mask sum: {train_mask.sum()}")
        print(f"  Val mask sum: {val_mask.sum()}")
        print(f"  Test mask sum: {test_mask.sum()}")

        train_frames = frames[train_mask]
        val_frames = frames[val_mask]
        test_frames = frames[test_mask]

        print(f"\nFinal shapes:")
        print(f"  Train: {train_frames.shape}")
        print(f"  Val: {val_frames.shape}")
        print(f"  Test: {test_frames.shape}")

        # 验证分割结果
        if train_frames.shape[0] == 0:
            print("\nERROR: Train set is still empty after fix!")
            print("Debug information:")
            print(f"  Timestamps type: {type(timestamps_pd[0])}")
            print(f"  Train times type: {type(train_times[0])}")
            print(f"  Sample timestamp comparison:")
            for i in range(min(5, len(timestamps_pd))):
                ts = timestamps_pd[i]
                in_train = ts in train_times_set
                print(f"    {ts} in train_times: {in_train}")
        else:
            print("\nSuccess! Data split correctly.")

        # 9. 保存数据
        print("\nSaving data...")
        output_file = f'./{output_prefix}_data.npz'

        np.savez_compressed(
            output_file,
            train_frames=train_frames.astype(np.float32),
            val_frames=val_frames.astype(np.float32),
            test_frames=test_frames.astype(np.float32),
            feature_columns=self.feature_columns,
            turbine_positions=turbine_positions,
            power_channel_idx=self.feature_columns.index('Patv'),
            timestamps=timestamps,  # 保存时间戳供调试
            train_times=train_times,
            val_times=val_times,
            test_times=test_times
        )

        # 保存scalers
        scalers_file = f'./{output_prefix}_scalers.pkl'
        with open(scalers_file, 'wb') as f:
            pickle.dump(self.scalers, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\nData saved to:")
        print(f"  - {output_file}")
        print(f"  - {scalers_file}")

        # 保存元数据
        metadata = {
            'n_features': len(self.feature_columns),
            'feature_names': self.feature_columns,
            'power_channel_idx': self.feature_columns.index('Patv'),
            'n_turbines': len(turbine_positions),
            'grid_shape': (28, 6),
            'train_samples': len(train_frames),
            'val_samples': len(val_frames),
            'test_samples': len(test_frames)
        }

        metadata_file = f'{output_prefix}_metadata.pkl'
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  - {metadata_file}")

        print("\n" + "=" * 60)
        print("Preprocessing Complete!")
        print("=" * 60)

        return train_frames, val_frames, test_frames


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description='SDWPF Data Preprocessing')
    parser.add_argument('--input', type=str, default='./sdwpf_2001_2112_full.csv',
                        help='Input CSV file path')
    parser.add_argument('--output_prefix', type=str, default='sdwpf',
                        help='Output file prefix')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')

    args = parser.parse_args()

    start_time = time.time()

    preprocessor = SDWPFPreprocessor()
    train_frames, val_frames, test_frames = preprocessor.process(
        filepath=args.input,
        output_prefix=args.output_prefix,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    elapsed_time = time.time() - start_time
    print(f"\nProcessing time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    if train_frames.shape[0] > 0:
        print(f"\nSuccess! You can now run training with:")
    else:
        print("\nERROR: Data processing failed - empty datasets!")
