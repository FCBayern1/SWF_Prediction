"""
改进的训练脚本 - 只预测功率通道
参考CViTRNN论文的两阶段预测流程
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import logging

# 导入模型
from models.models import get_model, MODEL_REGISTRY


# ==================== Dataset ====================
class WindPowerDataset(Dataset):
    """风电预测数据集 - 只预测功率通道"""

    def __init__(self, frames, lookback=10, horizon=6, power_channel_idx=-1):
        """
        Args:
            frames: 预处理后的数据 (T, H, W, C)
            lookback: 历史序列长度
            horizon: 预测步数
            power_channel_idx: 功率通道索引（默认最后一个）
        """
        self.frames = torch.FloatTensor(frames)
        self.lookback = lookback
        self.horizon = horizon
        self.power_channel_idx = power_channel_idx

    def __len__(self):
        return len(self.frames) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        # 输入：完整的多通道历史数据
        x = self.frames[idx:idx + self.lookback]

        # 目标：只包含未来功率通道
        y_full = self.frames[idx + self.lookback:idx + self.lookback + self.horizon]
        y_power = y_full[..., self.power_channel_idx:self.power_channel_idx + 1]  # 保持维度

        return x, y_power


# ==================== 训练函数 ====================
def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch_x, batch_y in progress_bar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()

        # 前向传播
        predictions = model(batch_x, target_len=batch_y.shape[1])

        # 计算损失（只针对功率通道）
        loss = criterion(predictions, batch_y)

        # 反向传播
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, scaler=None):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Validating"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 预测
            predictions = model(batch_x, target_len=batch_y.shape[1])

            loss = criterion(predictions, batch_y)
            total_loss += loss.item()

            # 收集预测结果
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    # 合并所有预测
    all_preds = np.concatenate([p.reshape(-1) for p in all_preds])
    all_targets = np.concatenate([t.reshape(-1) for t in all_targets])

    # 如果提供了scaler，进行反归一化
    if scaler is not None:
        all_preds = scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
        all_targets = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()

    # 计算指标
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds) if np.var(all_preds) > 0 else -1

    return total_loss / len(dataloader), rmse, mae, r2


# ==================== 主训练函数 ====================
def train(args):
    """主训练函数"""

    # 设置日志
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{args.save_dir}/training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting training with arguments: {args}")

    # 加载数据
    logger.info("Loading data...")
    data = np.load(args.data_path, allow_pickle=True)

    train_frames = data['train_frames']
    val_frames = data['val_frames']

    # 获取数据形状
    T, H, W, C = train_frames.shape
    input_shape = (H, W, C)
    logger.info(f"Data shape: T={T}, H={H}, W={W}, C={C}")

    # 确定功率通道索引
    if 'feature_columns' in data:
        feature_columns = list(data['feature_columns'])
        if 'Patv' in feature_columns:
            power_channel_idx = feature_columns.index('Patv')
        else:
            power_channel_idx = -1  # 默认最后一个
        logger.info(f"Power channel index: {power_channel_idx}")
    else:
        power_channel_idx = -1

    # 创建数据集
    train_dataset = WindPowerDataset(
        train_frames, args.lookback, args.horizon, power_channel_idx
    )
    val_dataset = WindPowerDataset(
        val_frames, args.lookback, args.horizon, power_channel_idx
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 创建模型
    device = torch.device(args.device)

    # 根据模型类型设置参数
    model_kwargs = {
        'input_size': H * W * C,
        'output_channels': 1,  # 只输出功率
        'spatial_dims': (H, W)
    }

    if args.model in ['lstm', 'gru']:
        model_kwargs.update({
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        })
    elif args.model == 'transformer':
        model_kwargs.update({
            'd_model': args.d_model,
            'nhead': args.num_heads,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        })
    elif args.model == 'convlstm':
        model_kwargs.update({
            'input_channels': C,
            'hidden_channels': args.hidden_channels,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        })

    # 使用模型注册表创建模型
    model = get_model(args.model, **model_kwargs)
    model = model.to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {args.model}, Parameters: {total_params:,}")

    # 损失函数和优化器
    criterion = nn.MSELoss() if args.loss == 'mse' else nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True
    )

    # 训练循环
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        logger.info(f"\n========== Epoch {epoch + 1}/{args.epochs} ==========")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.clip_grad)
        train_losses.append(train_loss)

        # 验证
        val_loss, rmse, mae, r2 = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # 更新学习率
        scheduler.step(val_loss)

        # 记录指标
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, LR: {current_lr:.6f}")

        # 保存最佳模型
        if r2 > best_r2:
            best_r2 = r2
            best_val_loss = val_loss

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2},
                'args': vars(args),
                'input_shape': input_shape,
                'power_channel_idx': power_channel_idx
            }

            torch.save(checkpoint, f"{args.save_dir}/best_model.pth")
            logger.info(f"Saved best model (R²={r2:.4f})")

        # Early stopping
        if args.early_stopping and epoch > args.patience * 2:
            if all(val_losses[-i] <= val_losses[-args.patience] for i in range(1, args.patience)):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'o-', label='Train')
    plt.plot(epochs, val_losses, 's-', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{args.save_dir}/training_curves.png")
    plt.close()

    logger.info(f"\nTraining completed!")
    logger.info(f"Best R²: {best_r2:.4f}, Best Val Loss: {best_val_loss:.4f}")

    # 保存训练配置
    config = {
        'args': vars(args),
        'best_metrics': {
            'r2': best_r2,
            'val_loss': best_val_loss
        },
        'input_shape': input_shape,
        'power_channel_idx': power_channel_idx
    }

    with open(f"{args.save_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)

    return best_r2, best_val_loss


# ==================== 参数解析 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='Wind Power Forecasting Training')

    # Model
    parser.add_argument('--model', type=str, default='lstm',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture')

    # Data
    parser.add_argument('--data_path', type=str, default='/Users/joshua/SWF_Prediction/Data/improved_preprocessed_data.npz',
                        help='Path to preprocessed data')
    parser.add_argument('--lookback', type=int, default=10,
                        help='Number of lookback timesteps')
    parser.add_argument('--horizon', type=int, default=6,
                        help='Prediction horizon')

    # Model parameters (for RNN models)
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden size for RNN models')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Transformer specific
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')

    # ConvLSTM specific
    parser.add_argument('--hidden_channels', type=int, default=64,
                        help='Hidden channels for ConvLSTM')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping')

    # Loss
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'mae'],
                        help='Loss function')

    # Optimizer
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for scheduler/early stopping')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    parser.add_argument('--pin_memory', action='store_true',
                        help='Pin memory for data loader')

    # Save
    parser.add_argument('--save_dir', type=str, default='experiments',
                        help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')

    args = parser.parse_args()

    # 创建实验目录
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    args.save_dir = os.path.join(args.save_dir, args.experiment_name)

    return args


# ==================== 主函数 ====================
if __name__ == '__main__':
    args = parse_args()

    print("=" * 60)
    print(f"Training {args.model} model for wind power forecasting")
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # 检查数据文件
    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found!")
        print("Please run data preprocessing first.")
        exit(1)

    best_r2, best_loss = train(args)

    print("\n" + "=" * 60)
    print(f"Training completed!")
    print(f"Best R²: {best_r2:.4f}")
    print(f"Best Val Loss: {best_loss:.4f}")
    print(f"Results saved to: {args.save_dir}")
    print("=" * 60)