"""
Step 1: 数据清洗脚本
清理SDWPF数据集中的异常值
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def clean_sdwpf_data(input_file='sdwpf_2001_2112_full.csv',
                     output_file='sdwpf_cleaned.csv'):
    """
    清洗SDWPF数据集
    """
    print("=" * 60)
    print("Step 1: 数据清洗")
    print("=" * 60)

    # 1. 加载数据
    print("\n📁 加载原始数据...")
    df = pd.read_csv(input_file)
    print(f"原始数据形状: {df.shape}")
    print(f"原始负功率数量: {(df['Patv'] < 0).sum():,}")

    # 2. 清洗负功率
    print("\n🔧 清洗负功率...")
    # 策略：小负值（>-10）设为0，大负值设为NaN后插值
    small_negative = (df['Patv'] < 0) & (df['Patv'] > -10)
    large_negative = df['Patv'] <= -10

    df.loc[small_negative, 'Patv'] = 0
    df.loc[large_negative, 'Patv'] = np.nan

    print(f"  设为0: {small_negative.sum():,}")
    print(f"  设为NaN: {large_negative.sum():,}")

    # 3. 修正异常范围
    print("\n🔧 修正异常特征范围...")

    # 风向范围修正
    df['Wdir'] = df['Wdir'].apply(lambda x: ((x + 180) % 360) - 180 if pd.notna(x) else x)
    df['Ndir'] = df['Ndir'].apply(lambda x: x % 360 if pd.notna(x) and x >= 0 else x)
    df['Wdir_w'] = df['Wdir_w'].apply(lambda x: x % 360 if pd.notna(x) and x >= 0 else x)

    # 叶片角度修正
    for col in ['Pab1', 'Pab2', 'Pab3']:
        df[col] = df[col].clip(lower=-5, upper=95)

    # 温度异常值
    df.loc[df['Etmp'] < -50, 'Etmp'] = np.nan
    df.loc[df['Etmp'] > 50, 'Etmp'] = np.nan
    df.loc[df['Itmp'] < -50, 'Itmp'] = np.nan
    df.loc[df['Itmp'] > 80, 'Itmp'] = np.nan

    # 功率上限（1.5MW涡轮机）
    df.loc[df['Patv'] > 1500, 'Patv'] = 1500

    # 4. 插值处理缺失值
    print("\n🔧 插值处理缺失值...")

    # 按涡轮机分组插值
    for turbine_id in tqdm(df['TurbID'].unique(), desc="插值进度"):
        turbine_mask = df['TurbID'] == turbine_id
        turbine_data = df.loc[turbine_mask]

        # 对每个特征列插值
        for col in df.columns:
            if col not in ['TurbID', 'Tmstamp']:
                # 线性插值
                df.loc[turbine_mask, col] = turbine_data[col].interpolate(
                    method='linear', limit_direction='both'
                )

    # 5. 最终检查
    print("\n✅ 清洗完成!")
    print(f"清洗后数据形状: {df.shape}")
    print(f"剩余负功率: {(df['Patv'] < 0).sum()}")
    print(f"剩余缺失值: {df.isnull().sum().sum()}")

    # 6. 保存清洗后的数据
    df.to_csv(output_file, index=False)
    print(f"\n💾 清洗后的数据已保存: {output_file}")

    # 7. 生成清洗报告图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 功率分布
    ax = axes[0]
    df['Patv'].hist(bins=50, ax=ax)
    ax.set_title('Cleaned Power Distribution')
    ax.set_xlabel('Power (kW)')
    ax.set_ylabel('Frequency')

    # 风速分布
    ax = axes[1]
    df['Wspd'].hist(bins=50, ax=ax)
    ax.set_title('Wind Speed Distribution')
    ax.set_xlabel('Wind Speed (m/s)')

    # 温度分布
    ax = axes[2]
    df['T2m'].hist(bins=50, ax=ax)
    ax.set_title('Temperature Distribution')
    ax.set_xlabel('Temperature (°C)')

    plt.tight_layout()
    plt.savefig('data_cleaning_report.png')
    print("📊 清洗报告已保存: data_cleaning_report.png")

    return df

if __name__ == '__main__':
    # Step 1: 数据清洗
    if not os.path.exists('sdwpf_cleaned.csv'):
        clean_sdwpf_data()
    else:
        print("✅ 清洗后的数据已存在，跳过Step 1")