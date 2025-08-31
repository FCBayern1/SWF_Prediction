"""
model.py - Corrected SimpleLSTM Model
"""

import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    """简化的LSTM模型 - 修复版"""

    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # 输出投影层 - 从hidden_size映射回input_size
        self.output_proj = nn.Linear(hidden_size, input_size)

    def forward(self, x, target_len=6):
        batch_size, seq_len, H, W, C = x.shape

        # Flatten spatial dimensions
        x_flat = x.reshape(batch_size, seq_len, -1)  # (batch, seq, H*W*C)

        # LSTM forward pass for the input sequence
        lstm_out, (h_n, c_n) = self.lstm(x_flat)

        # Generate predictions autoregressively
        predictions = []
        hidden = (h_n, c_n)

        # Use the last output from the encoder as the initial decoder input
        # Project it back to input size
        last_hidden = lstm_out[:, -1:, :]  # (batch, 1, hidden_size)

        for _ in range(target_len):
            # Project hidden state to input dimension for next LSTM input
            decoder_input = self.output_proj(last_hidden)  # (batch, 1, input_size)

            # LSTM step
            lstm_out, hidden = self.lstm(decoder_input, hidden)

            # Store the output (project to input space)
            output = self.output_proj(lstm_out.squeeze(1))  # (batch, input_size)
            predictions.append(output.reshape(batch_size, H, W, C))

            # Use current output as next input
            last_hidden = lstm_out

        return torch.stack(predictions, dim=1)


class ConvLSTM(nn.Module):
    """ConvLSTM模型（备选方案）"""

    def __init__(self, input_channels, hidden_channels=32, spatial_size=(7, 8)):
        super().__init__()

        self.H, self.W = spatial_size
        self.hidden_channels = hidden_channels

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )

        # LSTM (处理时间序列)
        self.lstm = nn.LSTM(
            input_size=hidden_channels * self.H * self.W,
            hidden_size=hidden_channels * self.H * self.W // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels * self.H * self.W // 2, hidden_channels * self.H * self.W),
            nn.ReLU(),
        )

        self.final_conv = nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1)

    def forward(self, x, target_len=6):
        batch_size, seq_len, H, W, C = x.shape

        # Encode each time step
        encoded_seq = []
        for t in range(seq_len):
            # (batch, H, W, C) -> (batch, C, H, W)
            x_t = x[:, t].permute(0, 3, 1, 2)
            enc_t = self.encoder(x_t)
            # Flatten spatial dimensions
            enc_t = enc_t.reshape(batch_size, -1)  # (batch, hidden_channels * H * W)
            encoded_seq.append(enc_t)

        encoded_seq = torch.stack(encoded_seq, dim=1)  # (batch, seq_len, features)

        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(encoded_seq)

        # Generate predictions
        predictions = []
        hidden = (h_n, c_n)
        last_hidden = lstm_out[:, -1:, :]

        for _ in range(target_len):
            # LSTM step
            lstm_out, hidden = self.lstm(last_hidden, hidden)

            # Decode to spatial dimensions
            decoded = self.decoder(lstm_out.squeeze(1))
            decoded = decoded.reshape(batch_size, self.hidden_channels, H, W)

            # Final convolution to get back to input channels
            output = self.final_conv(decoded)
            output = output.permute(0, 2, 3, 1)  # (batch, H, W, C)
            predictions.append(output)

            last_hidden = lstm_out

        return torch.stack(predictions, dim=1)


class SimpleGRU(nn.Module):
    """简化的GRU模型（另一个备选）"""

    def __init__(self, input_size, hidden_size=256, num_layers=2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # GRU层（比LSTM更简单，参数更少）
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, input_size)
        )

    def forward(self, x, target_len=6):
        batch_size, seq_len, H, W, C = x.shape

        # Flatten spatial dimensions
        x_flat = x.reshape(batch_size, seq_len, -1)

        # GRU forward
        gru_out, h_n = self.gru(x_flat)

        # Generate predictions
        predictions = []
        hidden = h_n

        # Start with last encoder output
        decoder_input = self.output_layer(gru_out[:, -1:, :])

        for _ in range(target_len):
            # GRU step
            gru_out, hidden = self.gru(decoder_input, hidden)

            # Generate output
            output = self.output_layer(gru_out.squeeze(1))
            predictions.append(output.reshape(batch_size, H, W, C))

            # Prepare next input
            decoder_input = output.unsqueeze(1)

        return torch.stack(predictions, dim=1)


# 测试函数
def test_models():
    """测试所有模型是否能正常运行"""

    print("Testing models...")

    # 创建测试数据
    batch_size = 2
    seq_len = 10
    H, W, C = 7, 8, 16
    target_len = 6

    x = torch.randn(batch_size, seq_len, H, W, C)

    # 测试SimpleLSTM
    print("\n1. Testing SimpleLSTM...")
    model1 = SimpleLSTM(input_size=H*W*C, hidden_size=256, num_layers=2)
    output1 = model1(x, target_len=target_len)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output1.shape}")
    print(f"   Expected: ({batch_size}, {target_len}, {H}, {W}, {C})")
    assert output1.shape == (batch_size, target_len, H, W, C), "Shape mismatch!"
    print("   ✓ SimpleLSTM works!")

    # 测试ConvLSTM
    print("\n2. Testing ConvLSTM...")
    model2 = ConvLSTM(input_channels=C, hidden_channels=32, spatial_size=(H, W))
    output2 = model2(x, target_len=target_len)
    print(f"   Output shape: {output2.shape}")
    assert output2.shape == (batch_size, target_len, H, W, C), "Shape mismatch!"
    print("   ✓ ConvLSTM works!")

    # 测试SimpleGRU
    print("\n3. Testing SimpleGRU...")
    model3 = SimpleGRU(input_size=H*W*C, hidden_size=256, num_layers=2)
    output3 = model3(x, target_len=target_len)
    print(f"   Output shape: {output3.shape}")
    assert output3.shape == (batch_size, target_len, H, W, C), "Shape mismatch!"
    print("   ✓ SimpleGRU works!")

    print("\n✓ All models passed the test!")


if __name__ == "__main__":
    test_models()