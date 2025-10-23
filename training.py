import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
from datetime import datetime
import argparse
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from models.CViTRNN import ImprovedCViTRNN

class SDWPFDataset(Dataset):
    def __init__(self, frames, lookback=12, horizon=8, stride=1, augment=False):
        """
        Args:
            frames: numpy array (T, H, W, C)
            lookback: 历史序列长度（改为12，对应3小时）
            horizon: 预测序列长度（改为8，对应2小时）
            stride: 滑动步长
            augment: 是否使用数据增强
        """
        self.frames = torch.FloatTensor(frames)
        self.lookback = lookback
        self.horizon = horizon
        self.stride = stride
        self.augment = augment
        self.T = len(frames)

        # 计算有效样本数
        self.valid_indices = list(range(0, self.T - self.lookback - self.horizon + 1, stride))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        x = self.frames[start_idx:start_idx + self.lookback]
        warm_up_target = self.frames[start_idx + 1:start_idx + self.lookback]
        pred_target = self.frames[start_idx + self.lookback:start_idx + self.lookback + self.horizon]

        # 数据增强（仅训练时）
        if self.augment and torch.rand(1) < 0.3:
            # 添加小量噪声
            noise_std = 0.01
            x = x + torch.randn_like(x) * noise_std

        return x, warm_up_target, pred_target

def denormalize_predictions(predictions, scalers_dict, turbine_positions, power_channel_idx):
    """
    反归一化预测结果
    Args:
        predictions: (batch, horizon, H, W, C)
        scalers_dict: 归一化器字典
        turbine_positions: 涡轮机位置映射
        power_channel_idx: 功率通道索引
    Returns:
        denormalized predictions
    """
    batch_size, horizon, H, W, C = predictions.shape
    denormalized = predictions.clone()

    for turbine_id, (h, w) in turbine_positions.items():
        if h < H and w < W:
            scaler_key = f'turbine_{turbine_id}_Patv'
            if scaler_key in scalers_dict:
                scaler = scalers_dict[scaler_key]

                # 提取该涡轮机的功率预测值
                power_preds = predictions[:, :, h, w, power_channel_idx]  # (batch, horizon)

                # 反归一化
                original_shape = power_preds.shape
                power_preds_flat = power_preds.reshape(-1, 1)
                power_denorm = scaler.inverse_transform(power_preds_flat)
                power_denorm = power_denorm.reshape(original_shape)

                # 更新反归一化后的值
                denormalized[:, :, h, w, power_channel_idx] = torch.FloatTensor(power_denorm)

    return denormalized

def compute_detailed_metrics(predictions, targets, mask=None):
    """
    计算详细的评估指标
    Args:
        predictions: 预测值
        targets: 真实值  
        mask: 可选的mask，用于忽略某些位置
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    # 基础指标
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)

    # R²指标
    r2 = r2_score(targets, predictions)

    # MAPE指标（避免除零）
    mask_nonzero = np.abs(targets) > 1e-6
    if np.any(mask_nonzero):
        mape = np.mean(np.abs((targets[mask_nonzero] - predictions[mask_nonzero]) / targets[mask_nonzero])) * 100
    else:
        mape = float('inf')

    # 归一化指标
    rmse_norm = rmse / (np.max(targets) - np.min(targets) + 1e-8)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'RMSE_norm': rmse_norm
    }

def train_epoch(model, dataloader, optimizer, criterion, device, warm_up_weight=0.3):
    """改进的训练函数"""
    model.train()

    total_loss = 0
    total_warm_up_loss = 0
    total_pred_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, (batch_x, warm_up_target, pred_target) in enumerate(progress_bar):
        batch_x = batch_x.to(device)
        warm_up_target = warm_up_target.to(device)
        pred_target = pred_target.to(device)

        optimizer.zero_grad()

        # 前向传播
        warm_up_outputs, predictions = model(batch_x, target_len=pred_target.shape[1])

        # 计算损失
        if warm_up_outputs is not None and warm_up_outputs.shape[1] > 0:
            warm_up_loss = criterion(warm_up_outputs, warm_up_target)
        else:
            warm_up_loss = torch.tensor(0.0, device=device)

        pred_loss = criterion(predictions, pred_target)

        # 总损失（调整权重）
        loss = warm_up_weight * warm_up_loss + pred_loss

        # 反向传播
        loss.backward()

        # 梯度裁剪（更保守）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()

        # 统计
        total_loss += loss.item()
        if isinstance(warm_up_loss, torch.Tensor):
            total_warm_up_loss += warm_up_loss.item()
        total_pred_loss += pred_loss.item()

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pred': f'{pred_loss.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_warm_up = total_warm_up_loss / len(dataloader)
    avg_pred = total_pred_loss / len(dataloader)

    return avg_loss, avg_warm_up, avg_pred

@torch.no_grad()
def validate(model, dataloader, criterion, device, scalers_dict=None, 
             turbine_positions=None, power_channel_idx=-1, denormalize=True):
    """改进的验证函数，支持反归一化评估"""
    model.eval()

    total_loss = 0
    all_predictions = []
    all_targets = []
    all_predictions_denorm = []
    all_targets_denorm = []

    for batch_x, _, pred_target in tqdm(dataloader, desc="Validating"):
        batch_x = batch_x.to(device)
        pred_target = pred_target.to(device)

        # 前向传播
        _, predictions = model(batch_x, target_len=pred_target.shape[1])

        loss = criterion(predictions, pred_target)
        total_loss += loss.item()

        # 转移到CPU进行评估
        pred_cpu = predictions.cpu().numpy()
        target_cpu = pred_target.cpu().numpy()

        # 提取功率通道进行评估
        pred_power = pred_cpu[:, :, :, :, power_channel_idx]
        target_power = target_cpu[:, :, :, :, power_channel_idx]

        all_predictions.append(pred_power.reshape(-1))
        all_targets.append(target_power.reshape(-1))

        # 反归一化评估（如果提供了scalers）
        if denormalize and scalers_dict is not None and turbine_positions is not None:
            pred_denorm = denormalize_predictions(
                predictions.cpu(), scalers_dict, turbine_positions, power_channel_idx
            )
            target_denorm = denormalize_predictions(
                pred_target.cpu(), scalers_dict, turbine_positions, power_channel_idx
            )

            pred_denorm_power = pred_denorm[:, :, :, :, power_channel_idx].numpy()
            target_denorm_power = target_denorm[:, :, :, :, power_channel_idx].numpy()

            all_predictions_denorm.append(pred_denorm_power.reshape(-1))
            all_targets_denorm.append(target_denorm_power.reshape(-1))

    avg_loss = total_loss / len(dataloader)

    # 标准化数据评估
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    metrics_norm = compute_detailed_metrics(all_predictions, all_targets)

    # 反归一化数据评估
    if all_predictions_denorm:
        all_predictions_denorm = np.concatenate(all_predictions_denorm)
        all_targets_denorm = np.concatenate(all_targets_denorm)
        metrics_denorm = compute_detailed_metrics(all_predictions_denorm, all_targets_denorm)
    else:
        metrics_denorm = metrics_norm

    return avg_loss, metrics_norm, metrics_denorm

def setup_logging(save_dir):
    """设置日志"""
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{save_dir}/training.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_improved_cvitrnn(config):
    """改进的主训练函数"""

    # 设置保存目录
    save_dir = config.get('save_dir', f'experiments/improved_cvitrnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    logger = setup_logging(save_dir)

    logger.info("=" * 60)
    logger.info("Improved CViTRNN Training")
    logger.info("=" * 60)

    # 打印配置
    logger.info("\nConfiguration:")
    for key, value in config.items():
        if key != 'device':
            logger.info(f" {key}: {value}")

    # 加载数据
    logger.info("\nLoading preprocessed data...")
    data = np.load(config['data_path'], allow_pickle=True)

    train_frames = data['train_frames']
    val_frames = data['val_frames'] 
    test_frames = data['test_frames']
    power_channel_idx = data['power_channel_idx'].item()
    turbine_positions = data['turbine_positions'].item()

    logger.info(f" Train shape: {train_frames.shape}")
    logger.info(f" Val shape: {val_frames.shape}")
    logger.info(f" Test shape: {test_frames.shape}")
    logger.info(f" Power channel index: {power_channel_idx}")

    # 加载scalers
    scalers_dict = None
    scalers_path = config['data_path'].replace('_data.npz', '_scalers.pkl')
    if os.path.exists(scalers_path):
        with open(scalers_path, 'rb') as f:
            scalers_dict = pickle.load(f)
        logger.info(f" Loaded {len(scalers_dict)} scalers")

    # 获取数据维度
    T, H, W, C = train_frames.shape

    # 创建数据集（使用改进的参数）
    logger.info("\nCreating datasets...")
    train_dataset = SDWPFDataset(
        train_frames, 
        config['lookback'], 
        config['horizon'],
        stride=config.get('stride', 1),
        augment=True  # 训练集启用数据增强
    )
    val_dataset = SDWPFDataset(val_frames, config['lookback'], config['horizon'])
    test_dataset = SDWPFDataset(test_frames, config['lookback'], config['horizon'])

    # 创建数据加载器（增大batch size）
    batch_size = min(config['batch_size'], 64)  # 限制最大batch size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('num_workers', 2),  # 增加工作进程
        pin_memory=True,
        drop_last=True  # 丢弃最后不完整的batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True
    )

    logger.info(f" Train batches: {len(train_loader)}")
    logger.info(f" Val batches: {len(val_loader)}")
    logger.info(f" Test batches: {len(test_loader)}")

    # 设备配置
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"\nDevice: {device}")

    # 创建改进的模型
    logger.info("\nCreating improved model...")
    model_config = {
        'in_channels': C,
        'embed_dim': config['embed_dim'],
        'num_cells': config['num_cells'], 
        'num_heads': config['num_heads'],
        'num_encoder_layers': config['num_encoder_layers'],
        'spatial_size': (H, W),
        'dropout': config['dropout']
    }

    model = ImprovedCViTRNN(model_config).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f" Total parameters: {total_params:,}")
    logger.info(f" Trainable parameters: {trainable_params:,}")

    # 损失函数和优化器（改进的配置）
    criterion = nn.MSELoss()

    # 使用AdamW优化器，更好的权重衰减
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)  # 更适合Transformer的beta参数
    )

    # 改进的学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['lr'],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 前10%时间用于warmup
        anneal_strategy='cos'
    )

    # 训练循环
    best_r2 = -float('inf')
    best_metrics = {}
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    val_metrics_history = []

    logger.info("\n" + "=" * 60)
    logger.info("Starting improved training...")
    logger.info("=" * 60)

    for epoch in range(config['epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['epochs']}")
        logger.info("-" * 40)

        # 训练
        train_loss, warm_up_loss, pred_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            config.get('warm_up_weight', 0.3)  # 降低warm-up权重
        )
        train_losses.append(train_loss)

        # 验证
        val_loss, metrics_norm, metrics_denorm = validate(
            model, val_loader, criterion, device, 
            scalers_dict, turbine_positions, power_channel_idx, 
            denormalize=True
        )
        val_losses.append(val_loss)
        val_metrics_history.append(metrics_denorm)

        # 记录指标
        logger.info(f"Train Loss: {train_loss:.4f} (Warm-up: {warm_up_loss:.4f}, Pred: {pred_loss:.4f})")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Metrics (Normalized) - RMSE: {metrics_norm['RMSE']:.4f}, MAE: {metrics_norm['MAE']:.4f}, R²: {metrics_norm['R2']:.4f}")
        logger.info(f"Val Metrics (Denormalized) - RMSE: {metrics_denorm['RMSE']:.2f}, MAE: {metrics_denorm['MAE']:.2f}, R²: {metrics_denorm['R2']:.4f}")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型（基于反归一化的R²）
        current_r2 = metrics_denorm['R2']
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_metrics = metrics_denorm.copy()
            epochs_without_improvement = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'metrics_norm': metrics_norm,
                'metrics_denorm': metrics_denorm,
                'config': config,
                'model_config': model_config
            }

            torch.save(checkpoint, f"{save_dir}/best_model.pth")
            logger.info(f"[SAVED] New best model (R²: {current_r2:.4f})")
        else:
            epochs_without_improvement += 1

        # 更新学习率
        scheduler.step()

        # 定期保存checkpoint
        if (epoch + 1) % config.get('save_interval', 20) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{save_dir}/checkpoint_epoch_{epoch + 1}.pth")

        # 早停
        if epochs_without_improvement >= config['early_stopping_patience']:
            logger.info(f"\nEarly stopping triggered! No improvement for {epochs_without_improvement} epochs.")
            break

    # 测试评估
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on test set...")
    logger.info("=" * 60)

    # 加载最佳模型
    checkpoint = torch.load(f"{save_dir}/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_metrics_norm, test_metrics_denorm = validate(
        model, test_loader, criterion, device,
        scalers_dict, turbine_positions, power_channel_idx,
        denormalize=True
    )

    logger.info("\nTest Results (Normalized):")
    for metric, value in test_metrics_norm.items():
        logger.info(f" {metric}: {value:.4f}")

    logger.info("\nTest Results (Denormalized - Real Units):")
    for metric, value in test_metrics_denorm.items():
        if metric in ['RMSE', 'MAE']:
            logger.info(f" {metric}: {value:.2f} kW")
        else:
            logger.info(f" {metric}: {value:.4f}")

    # 保存训练曲线和结果
    save_training_results(save_dir, train_losses, val_losses, val_metrics_history, 
                         test_metrics_denorm, config, logger)

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best Val R²: {best_r2:.4f}")
    logger.info(f"Test R²: {test_metrics_denorm['R2']:.4f}")
    logger.info(f"Results saved to: {save_dir}")

    return test_metrics_denorm['R2'], test_metrics_denorm['RMSE']

def save_training_results(save_dir, train_losses, val_losses, val_metrics_history, 
                         test_metrics, config, logger):
    """保存训练结果和可视化"""

    # 提取R²历史
    val_r2_scores = [m['R2'] for m in val_metrics_history]
    val_rmse_scores = [m['RMSE'] for m in val_metrics_history]

    # 绘制训练曲线
    plt.figure(figsize=(20, 5))

    # 损失曲线
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Train', alpha=0.7)
    plt.plot(val_losses, label='Validation', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # R²曲线  
    plt.subplot(1, 4, 2)
    plt.plot(val_r2_scores, 'g-', alpha=0.7, label='Val R²')
    plt.axhline(y=test_metrics['R2'], color='r', linestyle='--', 
                label=f'Test R²: {test_metrics["R2"]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('R² Score Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # RMSE曲线
    plt.subplot(1, 4, 3)
    plt.plot(val_rmse_scores, 'b-', alpha=0.7, label='Val RMSE')
    plt.axhline(y=test_metrics['RMSE'], color='r', linestyle='--',
                label=f'Test RMSE: {test_metrics["RMSE"]:.2f}')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (kW)')
    plt.title('RMSE Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 最后50个epoch的细节
    plt.subplot(1, 4, 4)
    start_idx = max(0, len(train_losses) - 50)
    epochs_range = range(start_idx, len(train_losses))
    plt.plot(epochs_range, train_losses[start_idx:], 'b-', alpha=0.7, label='Train')
    plt.plot(epochs_range, val_losses[start_idx:], 'r-', alpha=0.7, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss (Last 50 Epochs)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 保存详细结果
    results = {
        'best_val_metrics': val_metrics_history[np.argmax(val_r2_scores)],
        'test_metrics': test_metrics,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_r2_scores': val_r2_scores,
            'val_rmse_scores': val_rmse_scores
        },
        'config': config,
        'total_epochs': len(train_losses)
    }

    with open(f"{save_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=4)

    logger.info(f"Training curves saved to: {save_dir}/training_curves.png")
    logger.info(f"Detailed results saved to: {save_dir}/results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Improved CViTRNN Model')

    # 数据参数
    parser.add_argument('--data_path', type=str, default='Data/sdwpf_paper_data.npz',
                        help='Path to preprocessed data')
    parser.add_argument('--lookback', type=int, default=12,
                        help='Lookback window size (3 hours)')
    parser.add_argument('--horizon', type=int, default=8,
                        help='Prediction horizon (2 hours)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Sliding window stride')

    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=256,  # 增大嵌入维度
                        help='Embedding dimension')
    parser.add_argument('--num_cells', type=int, default=3,   # 增加层数
                        help='Number of CViTRNN cells')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=4,  # 增加层数
                        help='Number of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.15,  # 稍微增加dropout
                        help='Dropout rate')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=150,    # 增加训练轮数
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=48, # 增大batch size
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,     # 降低学习率
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,  # 增加正则化
                        help='Weight decay')
    parser.add_argument('--warm_up_weight', type=float, default=0.3,  # 降低warm-up权重
                        help='Weight for warm-up loss')
    parser.add_argument('--early_stopping_patience', type=int, default=30,  # 增加耐心
                        help='Early stopping patience')

    # 其他参数
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loader workers')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='Save checkpoint interval')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Save directory')

    args = parser.parse_args()

    # 设置设备
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 设置保存目录
    if args.save_dir is None:
        args.save_dir = f'experiments/improved_cvitrnn_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # 转换为配置字典
    config = vars(args)

    # 训练改进的模型
    test_r2, test_rmse = train_improved_cvitrnn(config)

    print(f"\nFinal Test R²: {test_r2:.4f}")
    print(f"Final Test RMSE: {test_rmse:.2f} kW")
