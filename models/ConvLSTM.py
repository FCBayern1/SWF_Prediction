import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


