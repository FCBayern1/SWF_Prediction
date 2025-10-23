import torch
import torch.nn as nn
import math

class HybridEmbedding(nn.Module):
    """改进的混合嵌入层，增加了位置编码"""

    def __init__(self, in_channels, embed_dim, spatial_size, max_seq_len=100):
        super().__init__()
        self.C = in_channels
        self.H, self.W = spatial_size
        self.embed_dim = embed_dim

        # 线性投影
        self.linear_proj = nn.Linear(self.H * self.W, embed_dim)

        # 可学习的空间嵌入
        self.spatial_embeddings = nn.Parameter(
            torch.randn(in_channels, embed_dim) * 0.02
        )

        # 时间位置编码
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(max_seq_len, embed_dim) * 0.02
        )

        # 空间位置编码
        self.spatial_pos_encoding = nn.Parameter(
            torch.randn(self.H, self.W, embed_dim) * 0.02
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, time_step=0):
        """
        Args:
            x: (batch, H, W, C)
            time_step: current time step for temporal position encoding
        Returns:
            embedded: (batch, C, embed_dim)
        """
        batch_size = x.shape[0]

        # 重排维度: (batch, H, W, C) -> (batch, C, H*W)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(batch_size, self.C, -1)

        # 变量嵌入
        var_embeddings = self.linear_proj(x)

        # 添加空间嵌入
        spatial_emb = self.spatial_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        hybrid_embeddings = var_embeddings + spatial_emb

        # 添加时间位置编码
        if time_step < self.temporal_pos_encoding.size(0):
            temporal_emb = self.temporal_pos_encoding[time_step].unsqueeze(0).unsqueeze(0)
            temporal_emb = temporal_emb.expand(batch_size, self.C, -1)
            hybrid_embeddings = hybrid_embeddings + temporal_emb

        # Layer normalization和dropout
        output = self.layer_norm(hybrid_embeddings)
        output = self.dropout(output)

        return output

class SpatioTemporalAttention(nn.Module):
    """专门针对时空数据的注意力机制"""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 多头注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 空间注意力权重
        self.spatial_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # 标准多头注意力
        attn_output, _ = self.multihead_attn(query, key, value)

        # 空间注意力权重
        spatial_weights = self.spatial_attention(attn_output)
        attn_output = attn_output * spatial_weights

        # 残差连接和层归一化
        output = self.layer_norm(query + self.dropout(attn_output))

        return output

class ImprovedCViTRNNCell(nn.Module):
    """改进的CViTRNN单元，增加残差连接和更好的注意力机制"""

    def __init__(self, embed_dim, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # 输入投影
        self.input_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        # 时空注意力
        self.spatiotemporal_attention = SpatioTemporalAttention(
            embed_dim, num_heads, dropout
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # 门控机制
        self.forget_gate = nn.Linear(embed_dim, embed_dim)
        self.input_gate = nn.Linear(embed_dim, embed_dim)
        self.cell_gate = nn.Linear(embed_dim, embed_dim)
        self.output_gate = nn.Linear(embed_dim, embed_dim)

        # 输出归一化
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_t, h_prev, c_prev):
        """
        Args:
            input_t: (batch, C, E)
            h_prev: (batch, C, E)  
            c_prev: (batch, C, E)
        """
        batch_size, C, E = input_t.shape

        # 连接输入和前一隐状态
        combined = torch.cat([input_t, h_prev], dim=-1)

        # 投影和归一化
        z1 = self.input_norm(self.input_proj(combined))

        # 时空注意力
        attn_output = self.spatiotemporal_attention(z1, z1, z1)

        # 前馈网络，残差连接
        ff_output = self.feed_forward(attn_output)
        z2 = self.output_norm(attn_output + ff_output)

        # LSTM风格的门控机制
        f_t = torch.sigmoid(self.forget_gate(z2))  # 遗忘门
        i_t = torch.sigmoid(self.input_gate(z2))   # 输入门
        c_candidate = torch.tanh(self.cell_gate(z2))  # 候选值
        o_t = torch.sigmoid(self.output_gate(z2))  # 输出门

        # 更新细胞状态和隐藏状态
        c_new = f_t * c_prev + i_t * c_candidate
        h_new = o_t * torch.tanh(c_new)

        return h_new, c_new

class ImprovedCViTRNN(nn.Module):
    """改进的CViTRNN模型"""

    def __init__(self, config):
        super().__init__()
        self.C = config['in_channels']
        self.embed_dim = config['embed_dim']
        self.num_cells = config['num_cells']
        self.H, self.W = config['spatial_size']

        # 混合嵌入层
        self.hybrid_embedding = HybridEmbedding(
            self.C, self.embed_dim, (self.H, self.W)
        )

        # CViTRNN单元
        self.cells = nn.ModuleList([
            ImprovedCViTRNNCell(
                self.embed_dim,
                config['num_heads'],
                config['num_encoder_layers'],
                config.get('dropout', 0.1) if i < self.num_cells - 1 else 0.0
            )
            for i in range(self.num_cells)
        ])

        # 输出投影
        self.output_proj = nn.Linear(self.embed_dim, self.H * self.W)

        # 输出激活函数（确保输出范围）
        self.output_activation = nn.Tanh()

        # 残差连接的投影层
        self.residual_proj = nn.Linear(self.H * self.W * self.C, self.H * self.W * self.C)

    def forward(self, x, target_len=6):
        """
        Args:
            x: (batch, seq_len, H, W, C)
            target_len: 预测步数
        Returns:
            warm_up_outputs: (batch, seq_len-1, H, W, C)
            predictions: (batch, target_len, H, W, C)
        """
        batch_size, seq_len, H, W, C = x.shape
        device = x.device

        # 初始化状态
        h_states = [torch.zeros(batch_size, self.C, self.embed_dim, device=device)
                   for _ in range(self.num_cells)]
        c_states = [torch.zeros(batch_size, self.C, self.embed_dim, device=device)
                   for _ in range(self.num_cells)]

        warm_up_outputs = []

        # Phase 1: Warm-up
        for t in range(seq_len):
            x_t = x[:, t]  # (batch, H, W, C)
            x_embed = self.hybrid_embedding(x_t, time_step=t)  # (batch, C, E)

            cell_input = x_embed
            for i, cell in enumerate(self.cells):
                h_states[i], c_states[i] = cell(cell_input, h_states[i], c_states[i])
                cell_input = h_states[i]

            if t > 0:
                # 重建所有通道
                final_hidden = h_states[-1]  # (batch, C, E)
                all_channels = []

                for c in range(self.C):
                    channel_pred = self.output_proj(final_hidden[:, c])
                    channel_pred = channel_pred.reshape(batch_size, H, W, 1)
                    all_channels.append(channel_pred)

                frame_pred = torch.cat(all_channels, dim=-1)

                # 残差连接（如果维度匹配）
                if t > 0:
                    residual = x[:, t-1]  # 前一帧作为残差
                    # 将residual reshape用于残差连接
                    residual_flat = residual.reshape(batch_size, -1)
                    frame_pred_flat = frame_pred.reshape(batch_size, -1)

                    # 残差连接
                    residual_projected = self.residual_proj(residual_flat)
                    frame_pred_flat = frame_pred_flat + residual_projected
                    frame_pred = frame_pred_flat.reshape(batch_size, H, W, C)

                # 应用激活函数
                frame_pred = self.output_activation(frame_pred)
                warm_up_outputs.append(frame_pred)

        # Phase 2: Prediction
        predictions = []

        for pred_step in range(target_len):
            final_hidden = h_states[-1]

            # 重建所有通道
            all_channels = []
            for c in range(self.C):
                channel_pred = self.output_proj(final_hidden[:, c])
                channel_pred = channel_pred.reshape(batch_size, H, W, 1)
                all_channels.append(channel_pred)

            frame_pred = torch.cat(all_channels, dim=-1)

            # 残差连接
            if len(predictions) > 0:
                residual = predictions[-1]  # 前一预测帧
            else:
                residual = warm_up_outputs[-1]  # 最后一个warm-up帧

            residual_flat = residual.reshape(batch_size, -1)
            frame_pred_flat = frame_pred.reshape(batch_size, -1)

            residual_projected = self.residual_proj(residual_flat)
            frame_pred_flat = frame_pred_flat + residual_projected
            frame_pred = frame_pred_flat.reshape(batch_size, H, W, C)

            # 应用激活函数
            frame_pred = self.output_activation(frame_pred)
            predictions.append(frame_pred)

            # 自回归更新
            x_embed = self.hybrid_embedding(frame_pred, time_step=seq_len+pred_step)
            cell_input = x_embed

            for i, cell in enumerate(self.cells):
                h_states[i], c_states[i] = cell(cell_input, h_states[i], c_states[i])
                cell_input = h_states[i]

        warm_up_outputs = torch.stack(warm_up_outputs, dim=1) if warm_up_outputs else None
        predictions = torch.stack(predictions, dim=1)

        return warm_up_outputs, predictions
