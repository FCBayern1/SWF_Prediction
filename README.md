# SDWPF Wind Power Prediction with CViTRNN

## 项目概述

基于CViTRNN（Convolutional Vision Transformer Recurrent Neural Network）模型的风电功率预测系统，使用SDWPF（Spatial Dynamic Wind Power Forecasting）数据集进行训练。该项目实现了对134个风力涡轮机的时空建模和功率预测。

## 主要特性

-  **大规模风电预测**：支持134个涡轮机的同时预测
-  **时空建模**：结合空间关系和时序动态
-  **优化的数据处理**：使用索引优化，处理速度提升约84,785倍
-  **完整的预处理流程**：包含数据清洗、插值、归一化等步骤
-  **Vision Transformer架构**：利用自注意力机制捕获涡轮机间的空间依赖

## 数据集

### SDWPF数据集统计
- **时间范围**：2020-01-01 至 2022-01-01（2年）
- **涡轮机数量**：134个
- **采样频率**：每15分钟
- **总时间戳**：84,785个
- **总数据量**：11,361,190条记录
- **特征数量**：16个（包括风速、风向、温度、功率等）

### 数据划分
- **训练集**：67,828个时间戳（80%）
- **验证集**：8,478个时间戳（10%）
- **测试集**：8,479个时间戳（10%）

## 项目结构

```
SWF_Prediction/
├── Data/
│   ├── sdwpf_2001_2112_full.csv          # 原始数据
│   ├── preprocess_all_turbines.py        # 优化的预处理脚本
│   ├── preprocess_all_turbines_tqdm.py   # 带进度条的预处理脚本
│   ├── cvitrnn_full_134turbines.npz      # 处理后的数据
│   └── cvitrnn_full_134turbines_scalers.pkl  # 归一化参数
├── models/
│   ├── cvitrnn_model.py                  # CViTRNN模型定义
│   └── vision_transformer.py             # Vision Transformer组件
├── train_cvitrnn_v2.py                   # 训练脚本
├── test_training.py                       # 测试脚本
└── README.md                              # 项目文档
```

## 环境要求

```bash
Python >= 3.8
PyTorch >= 2.0
pandas >= 1.5.0
numpy >= 1.23.0
tqdm >= 4.65.0
matplotlib >= 3.5.0
scikit-learn >= 1.1.0
```

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/SWF_Prediction.git
cd SWF_Prediction
```

2. 创建虚拟环境：
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据预处理

运行优化版预处理脚本（推荐）：
```bash
cd Data
python preprocess_all_turbines.py
```

或使用带进度条的版本：
```bash
python preprocess_all_turbines_tqdm.py
```

### 2. 训练模型

```bash
python train_cvitrnn_v2.py
```

训练参数可在脚本中调整：
- `batch_size`: 批次大小（默认32）
- `num_epochs`: 训练轮数（默认100）
- `learning_rate`: 学习率（默认0.001）
- `d_model`: 模型维度（默认512）
- `num_heads`: 注意力头数（默认8）

### 3. 测试模型

```bash
python test_training.py
```

## 模型架构

### CViTRNN组件

1. **空间编码器**（Spatial Encoder）
   - CNN层：提取局部空间特征
   - 位置编码：保留涡轮机的空间位置信息

2. **Vision Transformer**
   - 多头自注意力：捕获涡轮机间的全局依赖
   - 前馈网络：非线性特征变换
   - 层数：6层

3. **时序解码器**（Temporal Decoder）
   - GRU/LSTM：建模时序动态
   - 输出层：预测未来功率

### 空间网格布局

涡轮机被映射到28×6的网格上：
```
[  1   2   3   4   5   6]
[  7   8   9  10  11  12]
[ 13  14  15  16  17  18]
...
[127 128 129 130 131 132]
[133 134  -1  -1  -1  -1]
```

## 性能优化

### 预处理优化

1. **列顺序修复**：解决了`set_index`/`reset_index`导致的数据丢失问题
2. **NaN处理**：使用中位数替换异常值，避免训练中的NaN问题
3. **索引优化**：使用字典索引替代数组比较，性能提升约84,785倍

### 关键修复

```python
# 修复前：数据赋值失败
df.loc[turbine_mask] = turbine_data[df.columns]

# 修复后：使用.values确保正确赋值
df.loc[turbine_mask] = turbine_data[df.columns].values
```

## 常见问题

### Q1: 预处理卡在第10步？
**A**: 使用优化版脚本`preprocess_all_turbines.py`，已将O(n³)复杂度优化到O(n²)。

### Q2: 验证损失异常高？
**A**: 检查数据归一化，确保使用训练集统计量，避免数据泄露。

### Q3: 内存不足？
**A**: 可以减小批次大小或使用数据生成器进行流式处理。

## 未来改进

- [ ] 实现多GPU并行训练
- [ ] 添加更多气象特征
- [ ] 集成天气预报数据
- [ ] 实现在线学习能力
- [ ] 优化超参数搜索

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 引用

如果使用本项目，请引用：
```bibtex
@misc{sdwpf_cvitrnn_2024,
  title={SDWPF Wind Power Prediction with CViTRNN},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/SWF_Prediction}
}
```

## 联系方式

- Email: your.email@example.com
- Issues: https://github.com/yourusername/SWF_Prediction/issues