"""
SimpleLSTM.py - Corrected SimpleLSTM Model
"""

import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):

    def __init__(self, input_size, hidden_size=256, num_layers=3,
                 output_channels=1, dropout=0.2, spatial_dims=(7, 8)):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channels = output_channels
        self.H, self.W = spatial_dims

        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
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
        lstm_out, (h_n, c_n) = self.lstm(x_flat)

        # Phase 2: Prediction (自回归预测)
        predictions = []
        hidden = (h_n, c_n)

        # 使用最后的隐状态开始预测
        decoder_input = lstm_out[:, -1:, :]  # (batch, 1, hidden)

        for _ in range(target_len):
            # LSTM步进
            lstm_out, hidden = self.lstm(decoder_input, hidden)

            # 只输出功率通道
            power_output = self.output_proj(lstm_out.squeeze(1))
            power_output = power_output.reshape(batch_size, H, W, self.output_channels)
            predictions.append(power_output)

            # 准备下一步输入
            decoder_input = lstm_out

        return torch.stack(predictions, dim=1)


