import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class TransformerModel(nn.Module):
    """Transformer model for time series forecasting"""

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