"""
test_model.py
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle
import argparse

# 导入模型
from models.models import get_model


class WindPowerDataset(Dataset):
    """风电预测数据集 - 只预测功率通道"""

    def __init__(self, frames, lookback=10, horizon=6, power_channel_idx=-1):
        self.frames = torch.FloatTensor(frames)
        self.lookback = lookback
        self.horizon = horizon
        self.power_channel_idx = power_channel_idx

    def __len__(self):
        return len(self.frames) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        x = self.frames[idx:idx + self.lookback]
        y_full = self.frames[idx + self.lookback:idx + self.lookback + self.horizon]
        y_power = y_full[..., self.power_channel_idx:self.power_channel_idx + 1]
        return x, y_power


def load_scaler(scaler_path):
    """加载归一化器"""
    try:
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
        print(f"成功加载 {len(scalers)} 个scalers")

        # 获取第一个scaler作为功率通道的scaler
        first_scaler = list(scalers.values())[0]

        if hasattr(first_scaler, 'center_'):
            print(f"Scaler类型: RobustScaler")
            print(f"中位数: {first_scaler.center_[-1]:.2f}")
            print(f"IQR: {first_scaler.scale_[-1]:.2f}")

        return first_scaler
    except Exception as e:
        print(f"无法加载scaler: {e}")
        return None


def denormalize_power(data, scaler=None):
    """反归一化功率数据"""
    if scaler is not None and hasattr(scaler, 'inverse_transform'):
        # 使用实际的scaler
        n_features = scaler.center_.shape[0]
        n_samples = data.size

        # 创建完整特征数组（只关心最后一个特征）
        temp_data = np.zeros((n_samples, n_features))
        temp_data[:, -1] = data.flatten()

        # 反归一化
        denorm_data = scaler.inverse_transform(temp_data)[:, -1]
        return denorm_data.reshape(data.shape)
    else:
        # 使用估计参数
        print("使用默认反归一化参数...")
        median_estimate = 300  # MW
        iqr_estimate = 200  # MW

        denormalized = data * iqr_estimate + median_estimate
        denormalized = np.clip(denormalized, 0, 1500)

        return denormalized


def test_model(model_path, data_path, scaler_path=None):
    """测试模型性能"""

    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)

    # 加载模型配置和权重
    print("\n加载模型...")
    checkpoint = torch.load(model_path, map_location='cpu')

    # 提取配置
    if 'args' in checkpoint:
        args = checkpoint['args']
        print(f"模型配置:")
        print(f"  - Model: {args.get('model', 'lstm')}")
        print(f"  - Hidden size: {args.get('hidden_size', 256)}")
        print(f"  - Num layers: {args.get('num_layers', 3)}")
        print(f"  - Lookback: {args.get('lookback', 10)}")
        print(f"  - Horizon: {args.get('horizon', 6)}")

    # 加载数据
    print("\n加载测试数据...")
    data = np.load(data_path, allow_pickle=True)
    test_frames = data['test_frames']
    T, H, W, C = test_frames.shape
    print(f"测试数据形状: {test_frames.shape}")

    # 获取功率通道索引
    power_channel_idx = checkpoint.get('power_channel_idx', -1)

    # 创建数据集
    lookback = args.get('lookback', 10) if 'args' in checkpoint else 10
    horizon = args.get('horizon', 6) if 'args' in checkpoint else 6

    test_dataset = WindPowerDataset(test_frames, lookback, horizon, power_channel_idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"测试样本数: {len(test_dataset)}")

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 获取模型类型
    model_type = args.get('model', 'lstm') if 'args' in checkpoint else 'lstm'

    # 根据模型类型设置参数
    model_kwargs = {
        'input_size': H * W * C,
        'output_channels': 1,
        'spatial_dims': (H, W)
    }

    if model_type in ['lstm', 'gru']:
        model_kwargs.update({
            'hidden_size': args.get('hidden_size', 256) if 'args' in checkpoint else 256,
            'num_layers': args.get('num_layers', 3) if 'args' in checkpoint else 3,
            'dropout': 0  # 测试时不用dropout
        })
    elif model_type == 'transformer':
        model_kwargs.update({
            'd_model': args.get('d_model', 256) if 'args' in checkpoint else 256,
            'nhead': args.get('num_heads', 8) if 'args' in checkpoint else 8,
            'num_layers': args.get('num_layers', 3) if 'args' in checkpoint else 3,
            'dropout': 0
        })
    elif model_type == 'convlstm':
        model_kwargs.update({
            'input_channels': C,
            'hidden_channels': args.get('hidden_channels', 64) if 'args' in checkpoint else 64,
            'num_layers': args.get('num_layers', 3) if 'args' in checkpoint else 3,
            'dropout': 0
        })

    # 创建模型
    model = get_model(model_type, **model_kwargs)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"模型加载成功，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 加载scaler
    scaler = load_scaler(scaler_path) if scaler_path else None

    # 测试
    print("\n开始评估...")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="测试进度"):
            inputs = inputs.to(device)

            # 模型预测
            outputs = model(inputs, target_len=targets.shape[1])

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    # 合并结果 - 正确处理维度
    # all_preds: list of (batch, horizon, H, W, 1)
    # 转换为 (n_samples, horizon, H*W)
    predictions_norm = []
    targets_norm = []

    for pred, target in zip(all_preds, all_targets):
        # pred: (batch, horizon, H, W, 1)
        # target: (batch, horizon, H, W, 1)
        batch_size = pred.shape[0]
        pred_reshaped = pred.reshape(batch_size, horizon, -1)  # (batch, horizon, H*W)
        target_reshaped = target.reshape(batch_size, horizon, -1)
        predictions_norm.append(pred_reshaped)
        targets_norm.append(target_reshaped)

    predictions_norm = np.vstack(predictions_norm)  # (n_samples, horizon, H*W)
    targets_norm = np.vstack(targets_norm)

    # 只保留前50个涡轮机（活跃的）
    predictions_norm = predictions_norm[:, :, :50]
    targets_norm = targets_norm[:, :, :50]

    print(f"\n归一化空间统计:")
    print(f"  预测形状: {predictions_norm.shape}")
    print(f"  预测范围: [{predictions_norm.min():.3f}, {predictions_norm.max():.3f}]")
    print(f"  目标范围: [{targets_norm.min():.3f}, {targets_norm.max():.3f}]")

    # 在归一化空间评估
    r2_norm = r2_score(targets_norm.flatten(), predictions_norm.flatten())
    print(f"  归一化R²: {r2_norm:.4f}")

    # 反归一化
    print("\n应用反归一化...")
    predictions = denormalize_power(predictions_norm, scaler)
    targets = denormalize_power(targets_norm, scaler)

    print(f"反归一化后:")
    print(f"  预测范围: [{predictions.min():.1f}, {predictions.max():.1f}] MW")
    print(f"  目标范围: [{targets.min():.1f}, {targets.max():.1f}] MW")

    # 计算每个预测步的指标
    print("\n" + "=" * 60)
    print("分步评估结果:")
    print("=" * 60)

    step_metrics = []
    for h in range(horizon):
        pred_h = predictions[:, h, :].flatten()
        target_h = targets[:, h, :].flatten()

        mse_h = mean_squared_error(target_h, pred_h)
        rmse_h = np.sqrt(mse_h)
        mae_h = mean_absolute_error(target_h, pred_h)
        r2_h = r2_score(target_h, pred_h)

        # MAPE（避免除零）
        mask = target_h > 10
        if mask.sum() > 0:
            mape_h = np.mean(np.abs((target_h[mask] - pred_h[mask]) / target_h[mask])) * 100
        else:
            mape_h = 0

        step_metrics.append({
            'step': h + 1,
            'rmse': rmse_h,
            'mae': mae_h,
            'r2': r2_h,
            'mape': mape_h
        })

        print(
            f"Step {h + 1} (t+{(h + 1) * 10}min): RMSE={rmse_h:.2f} MW, MAE={mae_h:.2f} MW, R²={r2_h:.4f}, MAPE={mape_h:.1f}%")

    # 计算总体指标
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    r2 = r2_score(targets.flatten(), predictions.flatten())

    mask = targets.flatten() > 10
    if mask.sum() > 0:
        mape = np.mean(np.abs((targets.flatten()[mask] - predictions.flatten()[mask]) / targets.flatten()[mask])) * 100
    else:
        mape = 0

    print("\n" + "=" * 60)
    print("总体评估结果:")
    print("=" * 60)
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE:  {mae:.2f} MW")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # 模型性能评价
    print(f"\n模型性能评价:")
    if r2 > 0.8:
        print("优秀 - 模型性能很好")
    elif r2 > 0.6:
        print("良好 - 模型性能不错")
    elif r2 > 0.4:
        print("一般 - 需要改进")
    else:
        print("较差 - 建议重新训练")

    # 可视化
    visualize_results(predictions, targets, step_metrics, horizon)

    # 保存结果
    results = {
        'predictions': predictions,
        'targets': targets,
        'overall_metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        },
        'step_metrics': step_metrics
    }

    # 保存评估结果
    save_dir = os.path.dirname(model_path)
    np.savez(f'{save_dir}/test_results.npz', **results)
    print(f"\n结果已保存到 {save_dir}/test_results.npz")

    return results


def visualize_results(predictions, targets, step_metrics, horizon):
    """可视化测试结果"""

    plt.figure(figsize=(18, 12))

    # 1. 分步R²变化
    plt.subplot(3, 3, 1)
    steps = [m['step'] for m in step_metrics]
    r2_scores = [m['r2'] for m in step_metrics]
    plt.plot(steps, r2_scores, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Prediction Step')
    plt.ylabel('R² Score')
    plt.title('R² Score vs Prediction Horizon')
    plt.grid(True, alpha=0.3)
    plt.ylim([min(r2_scores) * 0.95, 1.0])

    # 2. 分步RMSE变化
    plt.subplot(3, 3, 2)
    rmse_scores = [m['rmse'] for m in step_metrics]
    plt.plot(steps, rmse_scores, 's-', color='red', linewidth=2, markersize=8)
    plt.xlabel('Prediction Step')
    plt.ylabel('RMSE (MW)')
    plt.title('RMSE vs Prediction Horizon')
    plt.grid(True, alpha=0.3)

    # 3. 第一步预测散点图
    plt.subplot(3, 3, 3)
    sample_size = min(5000, len(targets[:, 0, :].flatten()))
    indices = np.random.choice(len(targets[:, 0, :].flatten()), sample_size, replace=False)
    plt.scatter(targets[:, 0, :].flatten()[indices],
                predictions[:, 0, :].flatten()[indices],
                alpha=0.3, s=5, c='blue')
    max_val = max(targets[:, 0, :].max(), predictions[:, 0, :].max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Ideal')
    plt.xlabel('Actual Power (MW)')
    plt.ylabel('Predicted Power (MW)')
    plt.title(f'Step 1 Predictions (R²={step_metrics[0]["r2"]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. 最后一步预测散点图
    plt.subplot(3, 3, 4)
    plt.scatter(targets[:, -1, :].flatten()[indices],
                predictions[:, -1, :].flatten()[indices],
                alpha=0.3, s=5, c='green')
    max_val = max(targets[:, -1, :].max(), predictions[:, -1, :].max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Ideal')
    plt.xlabel('Actual Power (MW)')
    plt.ylabel('Predicted Power (MW)')
    plt.title(f'Step {horizon} Predictions (R²={step_metrics[-1]["r2"]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. 误差分布（所有步）
    plt.subplot(3, 3, 5)
    errors = predictions.flatten() - targets.flatten()
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(x=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Prediction Error (MW)')
    plt.ylabel('Frequency')
    plt.title('Overall Error Distribution')
    plt.grid(True, alpha=0.3)

    # 6. 时间序列对比（3个涡轮机）
    plt.subplot(3, 3, 6)
    for i in range(min(3, predictions.shape[2])):
        plt.plot(targets[:50, 0, i], label=f'Actual T{i + 1}', alpha=0.7)
        plt.plot(predictions[:50, 0, i], '--', label=f'Pred T{i + 1}', alpha=0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('Power (MW)')
    plt.title('Time Series Comparison (Step 1)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 7. 涡轮机性能分布
    plt.subplot(3, 3, 7)
    turbine_r2 = []
    n_turbines = predictions.shape[2]
    for i in range(n_turbines):
        pred_t = predictions[:, :, i].flatten()
        target_t = targets[:, :, i].flatten()
        if target_t.std() > 0:
            r2_t = r2_score(target_t, pred_t)
            turbine_r2.append(r2_t)
        else:
            turbine_r2.append(0)

    colors = ['green' if r > 0.7 else 'orange' if r > 0.5 else 'red' for r in turbine_r2]
    plt.bar(range(1, n_turbines + 1), turbine_r2, color=colors)
    plt.xlabel('Turbine ID')
    plt.ylabel('R² Score')
    plt.title('Performance by Turbine')
    plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)

    # 8. 预测误差随时间变化
    plt.subplot(3, 3, 8)
    mae_per_step = [m['mae'] for m in step_metrics]
    mape_per_step = [m['mape'] for m in step_metrics]

    ax1 = plt.gca()
    ax1.plot(steps, mae_per_step, 'b-', label='MAE', linewidth=2)
    ax1.set_xlabel('Prediction Step')
    ax1.set_ylabel('MAE (MW)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(steps, mape_per_step, 'r-', label='MAPE', linewidth=2)
    ax2.set_ylabel('MAPE (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Error Metrics vs Prediction Horizon')
    ax1.grid(True, alpha=0.3)

    # 9. 多步预测对比
    plt.subplot(3, 3, 9)
    # 选择一个样本展示多步预测
    sample_idx = 0
    turbine_idx = 0

    actual_sequence = targets[sample_idx, :, turbine_idx]
    pred_sequence = predictions[sample_idx, :, turbine_idx]

    x_axis = np.arange(1, horizon + 1) * 10  # 转换为分钟

    plt.plot(x_axis, actual_sequence, 'o-', label='Actual', linewidth=2)
    plt.plot(x_axis, pred_sequence, 's-', label='Predicted', linewidth=2)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Power (MW)')
    plt.title(f'Multi-step Prediction Example (Turbine 1)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('Wind Power Multi-Step Prediction Evaluation', fontsize=16, y=1.02)
    plt.tight_layout()

    # 保存图像
    plt.savefig('multi_step_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n可视化结果已保存到 multi_step_evaluation.png")


# ==================== 主函数 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test wind power forecasting model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data_path', type=str,
                        default='/Users/joshua/SWF_Prediction/Data/improved_preprocessed_data.npz',
                        help='Path to preprocessed data')
    parser.add_argument('--scaler_path', type=str,
                        default='/Users/joshua/SWF_Prediction/Data/improved_scalers.pkl',
                        help='Path to scalers')

    args = parser.parse_args()

    # 检查文件
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        exit(1)

    if not os.path.exists(args.data_path):
        print(f"Error: Data file {args.data_path} not found!")
        exit(1)

    # 运行测试
    results = test_model(args.model_path, args.data_path, args.scaler_path)