import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleGRU(nn.Module):
    """Simple GRU model (more efficient than LSTM)"""

    def __init__(self, input_size, hidden_size=256, num_layers=3,
                 output_channels=1, dropout=0.2, spatial_dims=(7, 8)):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channels = output_channels
        self.H, self.W = spatial_dims

        # GRU编码器
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出投影层
        self.output_proj = nn.Linear(hidden_size, self.H * self.W * output_channels)

    def forward(self, x, target_len=6):
        batch_size, seq_len, H, W, C = x.shape

        # 展平空间维度
        x_flat = x.reshape(batch_size, seq_len, -1)

        # Phase 1: Warm-up
        gru_out, h_n = self.gru(x_flat)

        # Phase 2: Prediction
        predictions = []
        hidden = h_n

        decoder_input = gru_out[:, -1:, :]

        for _ in range(target_len):
            gru_out, hidden = self.gru(decoder_input, hidden)

            power_output = self.output_proj(gru_out.squeeze(1))
            power_output = power_output.reshape(batch_size, H, W, self.output_channels)
            predictions.append(power_output)

            decoder_input = gru_out

        return torch.stack(predictions, dim=1)
