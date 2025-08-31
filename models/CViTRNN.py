import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ====================== Part 1: 混合嵌入层 ======================

class HybridEmbedding(nn.Module):
    """
    混合嵌入层：结合变量嵌入和空间嵌入
    将3D输入(H,W,C)转换为2D序列(C, embed_dim)
    """

    def __init__(self, in_channels, embed_dim, spatial_size):
        super().__init__()
        self.C = in_channels
        self.H, self.W = spatial_size
        self.embed_dim = embed_dim

        # 线性投影：将空间维度映射到嵌入维度
        self.linear_proj = nn.Linear(self.H * self.W, embed_dim)

        # 可学习的空间嵌入
        self.spatial_embeddings = nn.Parameter(
            torch.randn(in_channels, embed_dim) * 0.02
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, H, W, C)
        Returns:
            embedded: (batch, C, embed_dim)
        """
        batch_size = x.shape[0]

        # 重排维度: (batch, H, W, C) -> (batch, C, H*W)
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, C, H, W)
        x = x.reshape(batch_size, self.C, -1)  # (batch, C, H*W)

        # 变量嵌入
        var_embeddings = self.linear_proj(x)  # (batch, C, embed_dim)

        # 添加空间嵌入
        spatial_emb = self.spatial_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        hybrid_embeddings = var_embeddings + spatial_emb

        # Layer normalization
        output = self.layer_norm(hybrid_embeddings)

        return output


# ====================== Part 2: Transformer编码器 ======================

class TransformerEncoder(nn.Module):
    """
    标准Transformer编码器
    """

    def __init__(self, embed_dim, num_heads, num_layers, ff_dim=None, dropout=0.1):
        super().__init__()

        if ff_dim is None:
            ff_dim = embed_dim * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        """
        output = self.encoder(x)
        output = self.final_norm(output)
        return output


# ====================== Part 3: 简化LSTM ======================

class StreamlinedLSTM(nn.Module):
    """
    简化的LSTM（单门控）- 根据论文设计
    """

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()

        # 单个过滤门
        self.gate = nn.Linear(embed_dim, hidden_dim)
        # 候选值生成
        self.candidate = nn.Linear(embed_dim, hidden_dim)

    def forward(self, x, h_prev, c_prev):
        """
        Args:
            x: (batch, C, embed_dim) - Transformer输出
            h_prev: (batch, C, hidden_dim)
            c_prev: (batch, C, hidden_dim)
        """
        # 单个过滤门
        gate_output = torch.sigmoid(self.gate(x))

        # 候选值
        candidate_output = torch.tanh(self.candidate(x))

        # 更新cell state
        c_new = gate_output * (candidate_output + c_prev)

        # 更新hidden state
        h_new = gate_output * torch.tanh(c_new)

        return h_new, c_new


# ====================== Part 4: CViTRNN单元 ======================

class CViTRNNCell(nn.Module):
    """
    CViTRNN单元：混合嵌入 + Transformer + 简化LSTM
    """

    def __init__(self, in_channels, embed_dim, hidden_dim, spatial_size,
                 num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()

        self.C = in_channels
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # 混合嵌入层
        self.hybrid_embedding = HybridEmbedding(in_channels, embed_dim, spatial_size)

        # Transformer编码器
        self.transformer = TransformerEncoder(
            embed_dim, num_heads, num_layers, dropout=dropout
        )

        # 简化LSTM
        self.lstm = StreamlinedLSTM(embed_dim, hidden_dim)

        # 输入投影（用于多层堆叠）
        self.input_proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_data, h_prev, c_prev, is_first_cell=True):
        """
        前向传播
        """
        if is_first_cell and len(input_data.shape) == 4:
            # 第一个cell，处理原始输入 (batch, H, W, C)
            embedded = self.hybrid_embedding(input_data)
        else:
            # 后续cells，处理前一个cell的输出
            concatenated = torch.cat([input_data, h_prev], dim=-1)
            embedded = self.input_norm(self.input_proj(concatenated))

        # Transformer编码
        transformer_out = self.transformer(embedded)

        # 简化LSTM更新
        h_new, c_new = self.lstm(transformer_out, h_prev, c_prev)

        return h_new, c_new


# ====================== Part 5: 完整CViTRNN模型 ======================

class CViTRNN(nn.Module):
    """
    完整的CViTRNN模型
    """

    def __init__(self, config):
        super().__init__()

        # 配置参数
        self.num_cells = config.get('num_cells', 3)
        self.in_channels = config['in_channels']
        self.embed_dim = config.get('embed_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.H, self.W = config['spatial_size']
        self.spatial_size = config['spatial_size']

        # 堆叠多个CViTRNN单元
        self.cells = nn.ModuleList([
            CViTRNNCell(
                self.in_channels,
                self.embed_dim,
                self.hidden_dim,
                self.spatial_size,
                config.get('num_heads', 8),
                config.get('num_layers', 3),
                config.get('dropout', 0.1)
            ) for _ in range(self.num_cells)
        ])

        # 输出投影：从hidden_dim映射回空间维度
        self.output_proj = nn.Linear(self.hidden_dim, self.H * self.W)

        # 维度重建
        self.dimension_reconstruction = nn.Sequential(
            nn.Linear(self.in_channels * self.H * self.W,
                      self.in_channels * self.H * self.W),
            nn.LayerNorm(self.in_channels * self.H * self.W)
        )

    def forward(self, x, target_len=None):
        """
        Args:
            x: (batch, seq_len, H, W, C)
            target_len: 预测长度
        """
        batch_size, seq_len, H, W, C = x.shape
        device = x.device

        if target_len is None:
            target_len = 1

        # 初始化隐藏状态
        h_states = [
            torch.zeros(batch_size, C, self.hidden_dim).to(device)
            for _ in range(self.num_cells)
        ]
        c_states = [
            torch.zeros(batch_size, C, self.hidden_dim).to(device)
            for _ in range(self.num_cells)
        ]

        outputs = []

        # Phase 1: Warm-up阶段（处理输入序列）
        for t in range(seq_len - 1):
            input_t = x[:, t]  # (batch, H, W, C)

            # 通过所有CViTRNN单元
            for k in range(self.num_cells):
                if k == 0:
                    h_states[k], c_states[k] = self.cells[k](
                        input_t, h_states[k], c_states[k], is_first_cell=True
                    )
                else:
                    h_states[k], c_states[k] = self.cells[k](
                        h_states[k - 1], h_states[k], c_states[k], is_first_cell=False
                    )

            # 输出投影和维度重建
            output = self._reconstruct_output(h_states[-1], batch_size, H, W, C)
            outputs.append(output)

        # Phase 2: Prediction阶段（自回归预测）
        last_input = x[:, -1]  # 使用最后一个输入

        for _ in range(target_len):
            # 通过所有CViTRNN单元
            for k in range(self.num_cells):
                if k == 0:
                    h_states[k], c_states[k] = self.cells[k](
                        last_input, h_states[k], c_states[k], is_first_cell=True
                    )
                else:
                    h_states[k], c_states[k] = self.cells[k](
                        h_states[k - 1], h_states[k], c_states[k], is_first_cell=False
                    )

            # 输出投影和维度重建
            output = self._reconstruct_output(h_states[-1], batch_size, H, W, C)
            outputs.append(output)

            # 使用当前输出作为下一步输入（自回归）
            last_input = output

        # 返回预测部分
        return torch.stack(outputs[-target_len:], dim=1)

    def _reconstruct_output(self, hidden_state, batch_size, H, W, C):
        """
        从隐藏状态重建空间输出
        """
        # hidden_state: (batch, C, hidden_dim)

        # 投影到空间维度
        output = self.output_proj(hidden_state)  # (batch, C, H*W)

        # 重塑为空间格式
        output = output.reshape(batch_size, C, H, W)
        output = output.permute(0, 2, 3, 1)  # (batch, H, W, C)

        return output


# ====================== Part 6: 测试函数 ======================

def test_cvitrnn():
    """测试CViTRNN模型"""

    print("Testing CViTRNN Model...")
    print("-" * 50)

    # 配置
    config = {
        'num_cells': 3,
        'in_channels': 16,
        'embed_dim': 128,
        'hidden_dim': 128,
        'spatial_size': (7, 8),
        'num_heads': 8,
        'num_layers': 3,
        'dropout': 0.1
    }

    # 创建测试数据
    batch_size = 2
    seq_len = 10
    H, W = config['spatial_size']
    C = config['in_channels']
    target_len = 6

    x = torch.randn(batch_size, seq_len, H, W, C)

    # 创建模型
    model = CViTRNN(config)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model Configuration:")
    print(f"  Input shape: {x.shape}")
    print(f"  Spatial size: {H}×{W}")
    print(f"  Channels: {C}")
    print(f"  Embedding dim: {config['embed_dim']}")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Number of cells: {config['num_cells']}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # 前向传播
    print("\nRunning forward pass...")
    output = model(x, target_len=target_len)

    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, {target_len}, {H}, {W}, {C})")

    assert output.shape == (batch_size, target_len, H, W, C), "Shape mismatch!"
    print("\n✓ CViTRNN model works correctly!")

    return model


if __name__ == "__main__":
    model = test_cvitrnn()