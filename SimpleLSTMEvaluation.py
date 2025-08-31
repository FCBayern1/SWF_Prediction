"""
SimpleLSTM测试脚本 - 正确处理归一化数据
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from models.SimpleLSTM import SimpleLSTM

class SDWPFTestDataset(Dataset):
    def __init__(self, test_frames, sequence_length=10):
        self.frames = test_frames
        self.sequence_length = sequence_length
        self.T, self.H, self.W, self.C = test_frames.shape
        self.n_samples = self.T - sequence_length - 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.frames[idx:idx + self.sequence_length]
        y = self.frames[idx + self.sequence_length, :, :, -1]  # 功率通道
        return torch.FloatTensor(x), torch.FloatTensor(y)

def reverse_robust_scaling(data, feature_idx=-1):
    """
    反转RobustScaler的归一化
    对于SDWPF数据集，功率通常在0-1500MW范围
    """
    # 加载原始数据以获取真实范围（如果可能）
    try:
        # 尝试加载原始清洗后的数据
        import pandas as pd
        original_data = pd.read_csv('/Users/joshua/SWF_Prediction/Data/sdwpf_cleaned.csv')

        if 'Patv' in original_data.columns:
            # 获取功率的统计信息
            power_min = original_data['Patv'].min()
            power_max = original_data['Patv'].max()
            power_median = original_data['Patv'].median()
            power_q1 = original_data['Patv'].quantile(0.25)
            power_q3 = original_data['Patv'].quantile(0.75)

            print(f"原始功率统计：")
            print(f"  范围: [{power_min:.1f}, {power_max:.1f}] MW")
            print(f"  中位数: {power_median:.1f} MW")
            print(f"  IQR: [{power_q1:.1f}, {power_q3:.1f}] MW")

            # RobustScaler使用中位数和IQR进行缩放
            # scaled = (x - median) / IQR
            # 因此：x = scaled * IQR + median
            iqr = power_q3 - power_q1
            denormalized = data * iqr + power_median

            # 确保在合理范围内
            denormalized = np.clip(denormalized, 0, power_max)

            return denormalized
    except Exception as e:
        print(f"无法加载原始数据: {e}")

    # 备用方案：使用典型的风电场参数
    # 假设50个1.5MW涡轮机，功率范围0-1500MW
    print("使用默认反归一化参数...")

    # 基于数据范围[-0.35, 2.51]，假设这是RobustScaler的结果
    # RobustScaler: (x - median) / IQR
    # 假设median≈500MW, IQR≈400MW（典型值）

    median_estimate = 300  # MW
    iqr_estimate = 200     # MW

    denormalized = data * iqr_estimate + median_estimate
    denormalized = np.clip(denormalized, 0, 1500)  # 限制在0-1500MW

    return denormalized

def test_model():
    # 路径配置
    DATA_PATH = '/Users/joshua/SWF_Prediction/Data/improved_preprocessed_data.npz'
    MODEL_PATH = 'best_simple_model.pth'
    SCALER_PATH = '/Users/joshua/SWF_Prediction/Data/improved_scalers.pkl'

    # 尝试加载scalers
    scalers = None
    try:
        with open(SCALER_PATH, 'rb') as f:
            scalers = pickle.load(f)
        print(f"成功加载 {len(scalers)} 个scalers")

        # 检查scaler类型
        first_scaler = list(scalers.values())[0]
        print(f"Scaler类型: {type(first_scaler)}")

        # 如果是RobustScaler，获取其参数
        if hasattr(first_scaler, 'center_'):
            print(f"Scaler center (median): {first_scaler.center_[-1]:.2f}")  # 功率通道
            print(f"Scaler scale (IQR): {first_scaler.scale_[-1]:.2f}")
    except Exception as e:
        print(f"无法加载scalers: {e}")

    # 检查模型配置
    print("\n检查模型配置...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    else:
        state_dict = checkpoint

    hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
    num_layers = sum(1 for i in range(10) if f'lstm.weight_ih_l{i}' in state_dict)

    print(f"模型配置：hidden_size={hidden_size}, num_layers={num_layers}")

    # 加载数据
    print("\n加载数据...")
    data = np.load(DATA_PATH, allow_pickle=True)
    test_frames = data['test_frames']
    print(f"测试数据形状: {test_frames.shape}")

    # 检查数据范围
    power_channel = test_frames[:,:,:,-1]
    print(f"\n归一化功率数据统计：")
    print(f"  范围: [{power_channel.min():.3f}, {power_channel.max():.3f}]")
    print(f"  均值: {power_channel.mean():.3f}")
    print(f"  标准差: {power_channel.std():.3f}")

    # 创建数据集
    test_dataset = SDWPFTestDataset(test_frames, sequence_length=10)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 初始化模型
    print("\n初始化模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleLSTM(
        input_size=896,
        hidden_size=hidden_size,
        num_layers=num_layers
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"模型加载成功！总参数: {sum(p.numel() for p in model.parameters()):,}")

    # 测试
    print("\n开始测试...")
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="测试进度"):
            inputs = inputs.to(device)

            # 模型预测
            outputs = model(inputs, target_len=1)
            power_pred = outputs[:, 0, :, :, -1]

            all_preds.append(power_pred.cpu().numpy())
            all_targets.append(targets.numpy())

    # 合并结果
    predictions_norm = np.vstack(all_preds).reshape(-1, 56)[:, :50]
    targets_norm = np.vstack(all_targets).reshape(-1, 56)[:, :50]

    print(f"\n归一化结果：")
    print(f"  预测范围: [{predictions_norm.min():.3f}, {predictions_norm.max():.3f}]")
    print(f"  目标范围: [{targets_norm.min():.3f}, {targets_norm.max():.3f}]")

    # 在归一化空间评估
    r2_norm = r2_score(targets_norm.flatten(), predictions_norm.flatten())
    print(f"  归一化空间R²: {r2_norm:.4f}")

    # 反归一化
    print("\n应用反归一化...")

    if scalers and hasattr(list(scalers.values())[0], 'inverse_transform'):
        # 使用实际的scaler
        scaler = list(scalers.values())[0]

        # 准备反归一化
        n_features = 16
        n_samples = predictions_norm.size

        # 创建完整特征数组
        temp_pred = np.zeros((n_samples, n_features))
        temp_pred[:, -1] = predictions_norm.flatten()

        temp_target = np.zeros((n_samples, n_features))
        temp_target[:, -1] = targets_norm.flatten()

        # 反归一化
        pred_denorm = scaler.inverse_transform(temp_pred)[:, -1]
        target_denorm = scaler.inverse_transform(temp_target)[:, -1]

        predictions = pred_denorm.reshape(-1, 50)
        targets = target_denorm.reshape(-1, 50)
    else:
        # 使用估计的反归一化
        predictions = reverse_robust_scaling(predictions_norm)
        targets = reverse_robust_scaling(targets_norm)

    # 确保非负
    predictions = np.maximum(predictions, 0)
    targets = np.maximum(targets, 0)

    print(f"\n反归一化结果：")
    print(f"  预测范围: [{predictions.min():.1f}, {predictions.max():.1f}] MW")
    print(f"  目标范围: [{targets.min():.1f}, {targets.max():.1f}] MW")

    # 计算最终指标
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    r2 = r2_score(targets.flatten(), predictions.flatten())

    # 计算MAPE（避免除零）
    mask = targets.flatten() > 10  # 只对大于10MW的值计算MAPE
    if mask.sum() > 0:
        mape = np.mean(np.abs((targets.flatten()[mask] - predictions.flatten()[mask]) / targets.flatten()[mask])) * 100
    else:
        mape = 0

    print(f"\n" + "="*60)
    print(f"最终评估结果:")
    print(f"="*60)
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE:  {mae:.2f} MW")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # 性能评价
    print(f"\n模型性能评价:")
    if r2 > 0.8:
        print("✅ 优秀 - 模型性能很好")
    elif r2 > 0.6:
        print("👍 良好 - 模型性能不错")
    elif r2 > 0.4:
        print("⚠️ 一般 - 需要改进")
    else:
        print("❌ 较差 - 建议重新训练")

    # 可视化
    plt.figure(figsize=(16, 10))

    # 子图1：反归一化后的散点图
    plt.subplot(2, 3, 1)
    sample_size = min(5000, len(targets.flatten()))
    indices = np.random.choice(len(targets.flatten()), sample_size, replace=False)
    plt.scatter(targets.flatten()[indices],
                predictions.flatten()[indices],
                alpha=0.3, s=5, c='blue')

    # 添加理想线
    max_val = max(targets.max(), predictions.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Ideal')

    plt.xlabel('Actual Power (MW)', fontsize=12)
    plt.ylabel('Predicted Power (MW)', fontsize=12)
    plt.title(f'Predictions vs Actuals (R² = {r2:.3f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：误差分布
    plt.subplot(2, 3, 2)
    errors = predictions.flatten() - targets.flatten()
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(x=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Prediction Error (MW)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Error Distribution (MAE = {mae:.1f} MW)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 子图3：时间序列对比（前100个时间步）
    plt.subplot(2, 3, 3)
    for i in range(min(3, 50)):  # 显示3个涡轮机
        plt.plot(targets[:100, i], label=f'Actual T{i+1}', alpha=0.7)
        plt.plot(predictions[:100, i], '--', label=f'Pred T{i+1}', alpha=0.7)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Power (MW)', fontsize=12)
    plt.title('Time Series Comparison', fontsize=14)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # 子图4：相对误差分布
    plt.subplot(2, 3, 4)
    relative_errors = np.abs(errors[mask]) / targets.flatten()[mask] * 100
    plt.hist(relative_errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
    plt.xlabel('Relative Error (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Relative Error Distribution (MAPE = {mape:.1f}%)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 子图5：按涡轮机的性能
    plt.subplot(2, 3, 5)
    turbine_r2 = []
    for i in range(50):
        if targets[:, i].std() > 0:  # 避免常数列
            r2_i = r2_score(targets[:, i], predictions[:, i])
            turbine_r2.append(r2_i)
        else:
            turbine_r2.append(0)

    colors = ['green' if r > 0.7 else 'orange' if r > 0.5 else 'red' for r in turbine_r2]
    plt.bar(range(1, 51), turbine_r2, color=colors)
    plt.xlabel('Turbine ID', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Performance by Turbine', fontsize=14)
    plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)

    # 子图6：Q-Q图
    plt.subplot(2, 3, 6)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normality Check)', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.suptitle('Wind Power Prediction Model Evaluation', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('model_evaluation_complete.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n结果已保存到 model_evaluation_complete.png")

    return {
        'predictions': predictions,
        'targets': targets,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
    }

if __name__ == "__main__":
    results = test_model()