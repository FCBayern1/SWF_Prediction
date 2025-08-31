"""
SDWPF风电数据集的DataLoader完整实现
包含详细注释和使用示例
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from tqdm import tqdm


# ==================== Part 1: Dataset类的基础概念 ====================

class SDWPFDataset(Dataset):

    def __init__(self,
                 frames: np.ndarray,
                 sequence_length: int = 10,
                 prediction_horizon: int = 1,
                 target_feature_idx: int = 15,
                 transform: Optional[callable] = None):

        self.frames = frames
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_feature_idx = target_feature_idx
        self.transform = transform

        # 获取数据维度
        self.T, self.H, self.W, self.C = frames.shape

        # 计算可用的样本数量
        # 例如：总共1000个时间步，序列长度10，预测步长1
        # 则可用样本数 = 1000 - 10 - 1 + 1 = 990
        self.n_samples = self.T - sequence_length - prediction_horizon + 1

        if self.n_samples <= 0:
            raise ValueError(f"数据不足：需要至少 {sequence_length + prediction_horizon} 个时间步")

        print(f"✅ 数据集初始化成功!")
        print(f"   - 数据形状: {frames.shape}")
        print(f"   - 可用样本数: {self.n_samples}")
        print(f"   - 每个样本输入形状: ({sequence_length}, {self.H}, {self.W}, {self.C})")
        print(f"   - 每个样本输出形状: ({self.H}, {self.W})")

    def __len__(self) -> int:
        """
        返回数据集的大小
        DataLoader会调用这个方法来确定数据集有多少个样本
        """
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个数据样本
        DataLoader会调用这个方法来获取指定索引的数据

        Args:
            idx: 样本索引 (0 到 len(self)-1)

        Returns:
            x: 输入序列 (sequence_length, H, W, C)
            y: 目标值 (H, W) - 所有涡轮机的功率
        """
        # 1. 获取输入序列（历史数据）
        # 例如：idx=0时，获取frames[0:10]作为输入
        start_idx = idx
        end_idx = idx + self.sequence_length
        x = self.frames[start_idx:end_idx]  # (sequence_length, H, W, C)

        # 2. 获取目标值（要预测的未来值）
        # 例如：预测第11个时间步的功率
        target_idx = idx + self.sequence_length + self.prediction_horizon - 1
        y = self.frames[target_idx, :, :, self.target_feature_idx]  # (H, W)

        # 3. 应用数据变换（如果有）
        if self.transform:
            x = self.transform(x)

        # 4. 转换为PyTorch张量
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        return x, y


# ==================== Part 2: 扁平化版本的Dataset ====================

class SDWPFDatasetFlat(Dataset):
    """
    扁平化版本的数据集 - 适用于标准LSTM模型
    将空间维度展平，更适合你的SimpleLSTM模型
    """

    def __init__(self,
                 frames: np.ndarray,
                 sequence_length: int = 10,
                 prediction_horizon: int = 1):
        """
        初始化扁平化数据集

        与原版的区别：
        - 输入会被展平为 (sequence_length, H*W*C)
        - 输出会被展平为 (H*W,) 只包含功率值
        """
        self.frames = frames
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        self.T, self.H, self.W, self.C = frames.shape
        self.n_samples = self.T - sequence_length - prediction_horizon + 1

        # 扁平化后的维度
        self.input_dim = self.H * self.W * self.C  # 7*8*16 = 896
        self.output_dim = self.H * self.W  # 7*8 = 56 (但只有50个活跃)

        print(f"✅ 扁平化数据集初始化成功!")
        print(f"   - 输入维度: {self.input_dim}")
        print(f"   - 输出维度: {self.output_dim}")
        print(f"   - 样本数: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 获取输入序列
        x = self.frames[idx:idx + self.sequence_length]

        # 获取目标（功率值）
        target_idx = idx + self.sequence_length + self.prediction_horizon - 1
        y = self.frames[target_idx, :, :, -1]  # 假设功率是最后一个特征

        # 展平
        x_flat = x.reshape(self.sequence_length, -1)  # (seq_len, H*W*C)
        y_flat = y.flatten()  # (H*W,)

        return torch.FloatTensor(x_flat), torch.FloatTensor(y_flat)


# ==================== Part 3: 创建DataLoader ====================

def create_dataloaders(train_frames: np.ndarray,
                       val_frames: np.ndarray,
                       test_frames: np.ndarray,
                       sequence_length: int = 10,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       use_flat: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试的DataLoader

    DataLoader是什么？
    - DataLoader负责批量加载数据、打乱顺序、多进程加载等
    - 它将Dataset包装起来，提供迭代器接口

    Args:
        train_frames: 训练数据
        val_frames: 验证数据
        test_frames: 测试数据
        sequence_length: 序列长度
        batch_size: 批次大小（每次训练使用多少个样本）
        num_workers: 数据加载的进程数（0表示主进程加载）
        use_flat: 是否使用扁平化版本

    Returns:
        train_loader, val_loader, test_loader
    """

    print("\n" + "=" * 60)
    print("创建DataLoader")
    print("=" * 60)

    # 选择Dataset类
    DatasetClass = SDWPFDatasetFlat if use_flat else SDWPFDataset

    # 创建Dataset实例
    print("\n📦 创建训练集...")
    train_dataset = DatasetClass(train_frames, sequence_length)

    print("\n📦 创建验证集...")
    val_dataset = DatasetClass(val_frames, sequence_length)

    print("\n📦 创建测试集...")
    test_dataset = DatasetClass(test_frames, sequence_length)

    # 创建DataLoader
    print("\n🔄 创建DataLoader...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集需要打乱
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,  # GPU加速
        drop_last=True  # 丢弃最后不完整的批次
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"\n✅ DataLoader创建完成!")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 训练批次数: {len(train_loader)}")
    print(f"   - 验证批次数: {len(val_loader)}")
    print(f"   - 测试批次数: {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ==================== Part 4: 使用示例 ====================

def demonstrate_dataloader_usage():
    """
    演示DataLoader的使用方法
    """
    print("\n" + "=" * 60)
    print("DataLoader使用演示")
    print("=" * 60)

    # 1. 加载你的预处理数据
    print("\n1️⃣ 加载预处理数据...")
    data = np.load('improved_preprocessed_data.npz', allow_pickle=True)
    train_frames = data['train_frames']
    val_frames = data['val_frames']
    test_frames = data['test_frames']

    print(f"   训练数据形状: {train_frames.shape}")
    print(f"   验证数据形状: {val_frames.shape}")
    print(f"   测试数据形状: {test_frames.shape}")

    # 2. 创建DataLoader
    print("\n2️⃣ 创建DataLoader...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_frames,
        val_frames,
        test_frames,
        sequence_length=10,
        batch_size=32,
        num_workers=0,  # Windows下建议设为0
        use_flat=True  # 使用扁平化版本
    )

    # 3. 遍历DataLoader获取数据
    print("\n3️⃣ 从DataLoader获取数据...")
    print("-" * 40)

    # 获取一个批次的数据
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"批次 {batch_idx + 1}:")
        print(f"  输入形状: {inputs.shape}")  # (batch_size, seq_len, input_dim)
        print(f"  目标形状: {targets.shape}")  # (batch_size, output_dim)
        print(f"  输入数据类型: {inputs.dtype}")
        print(f"  设备: {inputs.device}")

        # 只展示第一个批次
        if batch_idx == 0:
            print(f"\n  第一个样本的统计信息:")
            print(f"    输入均值: {inputs[0].mean():.4f}")
            print(f"    输入标准差: {inputs[0].std():.4f}")
            print(f"    目标均值: {targets[0].mean():.4f}")
            print(f"    目标范围: [{targets[0].min():.2f}, {targets[0].max():.2f}]")

        break  # 只展示第一个批次

    # 4. 在训练循环中使用
    print("\n4️⃣ 训练循环示例...")
    print("-" * 40)

    # 模拟一个epoch的训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 简单的训练循环示例
    total_loss = 0
    num_batches = 5  # 只演示前5个批次

    print("\n模拟训练过程:")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        # 移动到GPU（如果可用）
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 这里通常会有：
        # 1. optimizer.zero_grad()
        # 2. outputs = model(inputs)
        # 3. loss = criterion(outputs, targets)
        # 4. loss.backward()
        # 5. optimizer.step()

        # 模拟损失
        fake_loss = torch.rand(1).item()
        total_loss += fake_loss

        print(f"  批次 {batch_idx + 1}/{num_batches} - 损失: {fake_loss:.4f}")

    avg_loss = total_loss / num_batches
    print(f"\n平均损失: {avg_loss:.4f}")

    return train_loader, val_loader, test_loader


# ==================== Part 5: 高级功能 ====================

class SDWPFDatasetAdvanced(Dataset):
    """
    高级版本的数据集 - 包含更多功能
    """

    def __init__(self,
                 frames: np.ndarray,
                 sequence_length: int = 10,
                 prediction_horizon: int = 1,
                 augment: bool = False,
                 noise_level: float = 0.01):
        """
        高级数据集，包含数据增强等功能

        Args:
            augment: 是否进行数据增强
            noise_level: 添加噪声的强度
        """
        self.frames = frames
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.augment = augment
        self.noise_level = noise_level

        self.T, self.H, self.W, self.C = frames.shape
        self.n_samples = self.T - sequence_length - prediction_horizon + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 基础数据获取
        x = self.frames[idx:idx + self.sequence_length].copy()
        target_idx = idx + self.sequence_length + self.prediction_horizon - 1
        y = self.frames[target_idx, :, :, -1].copy()

        # 数据增强（训练时）
        if self.augment:
            # 添加高斯噪声
            noise = np.random.normal(0, self.noise_level, x.shape)
            x = x + noise

            # 随机缩放
            scale = np.random.uniform(0.95, 1.05)
            x = x * scale

        # 展平
        x_flat = x.reshape(self.sequence_length, -1)
        y_flat = y.flatten()

        return torch.FloatTensor(x_flat), torch.FloatTensor(y_flat)


def collate_fn(batch):
    """
    自定义的批处理函数
    可以用于处理变长序列或特殊的批处理需求
    """
    inputs, targets = zip(*batch)

    # 堆叠成批次
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)

    # 可以在这里添加额外的处理
    # 例如：padding、masking等

    return inputs, targets


# ==================== Part 6: 性能测试 ====================

def benchmark_dataloader(loader: DataLoader, num_batches: int = 100):
    """
    测试DataLoader的性能
    """
    import time

    print("\n⏱️ DataLoader性能测试...")
    print("-" * 40)

    start_time = time.time()

    for i, (inputs, targets) in enumerate(loader):
        if i >= num_batches:
            break
        # 模拟数据传输到GPU
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

    elapsed_time = time.time() - start_time

    print(f"处理 {num_batches} 个批次用时: {elapsed_time:.2f} 秒")
    print(f"平均每批次: {elapsed_time / num_batches * 1000:.2f} 毫秒")
    print(f"吞吐量: {num_batches * loader.batch_size / elapsed_time:.0f} 样本/秒")


# ==================== 主函数 ====================

if __name__ == "__main__":
    import os

    # 检查数据文件是否存在
    if not os.path.exists('improved_preprocessed_data.npz'):
        print("❌ 错误：找不到预处理数据文件!")
        print("请先运行数据预处理脚本：python process_improve.py")
    else:
        # 运行演示
        train_loader, val_loader, test_loader = demonstrate_dataloader_usage()

        # 性能测试
        print("\n" + "=" * 60)
        print("性能测试")
        print("=" * 60)
        benchmark_dataloader(train_loader, num_batches=50)

        print("\n✅ 演示完成!")
        print("\n💡 提示：")
        print("  - 可以调整batch_size来控制内存使用")
        print("  - num_workers在Windows下建议设为0")
        print("  - 使用pin_memory=True可以加速GPU传输")
        print("  - shuffle=True确保训练数据的随机性")