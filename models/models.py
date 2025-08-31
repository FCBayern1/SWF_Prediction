"""
models.py - 风电功率预测模型定义
所有模型都只预测功率通道
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImprovedLSTM(nn.Module):
    """改进的LSTM - 只输出功率预测"""

    def __init__(self, input_size, hidden_size=256, num_layers=3,
                 output_channels=1, dropout=0.2, spatial_dims=(7, 8)):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channels = output_channels
        self.H, self.W = spatial_dims

        # 编码器LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 解码器LSTM（输入和隐藏维度相同）
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0  # 预测时不使用dropout
        )

        # 输出投影层 - 只投影到功率通道
        self.output_proj = nn.Linear(hidden_size, self.H * self.W * output_channels)

    def forward(self, x, target_len=6):
        """
        两阶段前向传播（参考CViTRNN）
        Args:
            x: (batch, seq_len, H, W, C)
            target_len: 预测步数
        Returns:
            predictions: (batch, target_len, H, W, 1) 只包含功率预测
        """
        batch_size, seq_len, H, W, C = x.shape

        # 展平空间维度
        x_flat = x.reshape(batch_size, seq_len, -1)

        # Phase 1: Warm-up (编码历史序列)
        encoder_out, (h_n, c_n) = self.encoder_lstm(x_flat)

        # Phase 2: Prediction (自回归预测)
        predictions = []
        hidden = (h_n, c_n)

        # 使用编码器的最后输出作为解码器的初始输入
        decoder_input = encoder_out[:, -1:, :]  # (batch, 1, hidden_size)

        for _ in range(target_len):
            # 解码器LSTM步进
            decoder_out, hidden = self.decoder_lstm(decoder_input, hidden)

            # 只输出功率通道
            power_output = self.output_proj(decoder_out.squeeze(1))
            power_output = power_output.reshape(batch_size, H, W, self.output_channels)
            predictions.append(power_output)

            # 使用当前输出作为下一步输入
            decoder_input = decoder_out

        return torch.stack(predictions, dim=1)


class ImprovedGRU(nn.Module):
    """改进的GRU - 只输出功率预测"""

    def __init__(self, input_size, hidden_size=256, num_layers=3,
                 output_channels=1, dropout=0.2, spatial_dims=(7, 8)):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channels = output_channels
        self.H, self.W = spatial_dims

        # 编码器GRU
        self.encoder_gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 解码器GRU
        self.decoder_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0
        )

        # 输出投影层
        self.output_proj = nn.Linear(hidden_size, self.H * self.W * output_channels)

    def forward(self, x, target_len=6):
        batch_size, seq_len, H, W, C = x.shape

        # 展平空间维度
        x_flat = x.reshape(batch_size, seq_len, -1)

        # Phase 1: Warm-up
        encoder_out, h_n = self.encoder_gru(x_flat)

        # Phase 2: Prediction
        predictions = []
        hidden = h_n

        decoder_input = encoder_out[:, -1:, :]

        for _ in range(target_len):
            decoder_out, hidden = self.decoder_gru(decoder_input, hidden)

            power_output = self.output_proj(decoder_out.squeeze(1))
            power_output = power_output.reshape(batch_size, H, W, self.output_channels)
            predictions.append(power_output)

            decoder_input = decoder_out

        return torch.stack(predictions, dim=1)


class TransformerPredictor(nn.Module):
    """Transformer预测器 - 只输出功率预测"""

    def __init__(self, input_size, d_model=256, nhead=8, num_layers=3,
                 output_channels=1, dropout=0.1, spatial_dims=(7, 8)):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.output_channels = output_channels
        self.H, self.W = spatial_dims

        # 输入投影
        self.input_proj = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 解码器
        self.decoder = nn.LSTM(d_model, d_model, batch_first=True)

        # 输出投影
        self.output_proj = nn.Linear(d_model, self.H * self.W * output_channels)

    def forward(self, x, target_len=6):
        batch_size, seq_len, H, W, C = x.shape

        # 展平并投影
        x_flat = x.reshape(batch_size, seq_len, -1)
        x_proj = self.input_proj(x_flat)

        # 添加位置编码
        x_pos = self.pos_encoder(x_proj)

        # Transformer编码
        memory = self.transformer(x_pos)

        # 自回归解码
        predictions = []
        decoder_input = memory[:, -1:, :]
        hidden = None

        for _ in range(target_len):
            decoder_out, hidden = self.decoder(decoder_input, hidden)

            power_output = self.output_proj(decoder_out.squeeze(1))
            power_output = power_output.reshape(batch_size, H, W, self.output_channels)
            predictions.append(power_output)

            decoder_input = decoder_out

        return torch.stack(predictions, dim=1)


class ConvLSTMCell(nn.Module):
    """ConvLSTM单元"""

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # 分割门
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


class ConvLSTMPredictor(nn.Module):
    """ConvLSTM预测器 - 保留空间结构"""

    def __init__(self, input_channels, hidden_channels=64, num_layers=3,
                 output_channels=1, dropout=0.2, spatial_dims=(7, 8)):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.output_channels = output_channels
        self.H, self.W = spatial_dims

        # ConvLSTM层
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_channels if i == 0 else hidden_channels,
                hidden_channels
            )
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

        # 输出卷积 - 只输出功率通道
        self.output_conv = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, x, target_len=6):
        batch_size, seq_len, H, W, C = x.shape

        # 调整维度 (batch, seq, C, H, W)
        x = x.permute(0, 1, 4, 2, 3)

        # 初始化隐状态
        h = [torch.zeros(batch_size, self.hidden_channels, H, W, device=x.device)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_channels, H, W, device=x.device)
             for _ in range(self.num_layers)]

        # Phase 1: Warm-up
        for t in range(seq_len):
            input_t = x[:, t]

            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(input_t, h[i], c[i])
                input_t = h[i]
                if self.dropout and i < self.num_layers - 1:
                    input_t = self.dropout(input_t)

        # Phase 2: Prediction
        predictions = []

        for _ in range(target_len):
            # 使用最后的隐状态预测
            power_output = self.output_conv(h[-1])
            predictions.append(power_output)

            # 自回归：将预测作为输入（这里简化处理）
            # 实际应用中可能需要更复杂的策略
            input_t = torch.zeros(batch_size, self.input_channels, H, W, device=x.device)
            input_t[:, -1:] = power_output  # 将功率预测放入最后一个通道

            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(input_t, h[i], c[i])
                input_t = h[i]
                if self.dropout and i < self.num_layers - 1:
                    input_t = self.dropout(input_t)

        # (batch, target_len, 1, H, W) -> (batch, target_len, H, W, 1)
        predictions = torch.stack(predictions, dim=1).permute(0, 1, 3, 4, 2)

        return predictions


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


# 模型注册表
MODEL_REGISTRY = {
    'lstm': ImprovedLSTM,
    'gru': ImprovedGRU,
    'transformer': TransformerPredictor,
    'convlstm': ConvLSTMPredictor
}


def get_model(model_name, **kwargs):
    """获取模型实例"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(**kwargs)