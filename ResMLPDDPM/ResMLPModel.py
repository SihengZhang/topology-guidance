import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, T, d_time):
        assert d_time % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_time, step=2) / d_time * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_time // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_time // 2, 2]
        emb = emb.view(T, d_time)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb

class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, time_embedding_dim):
        super().__init__()

        # First branch (processing the main input)
        self.groupnorm_1 = nn.GroupNorm(num_groups=1, num_channels=input_dim)
        self.silu_1 = nn.SiLU()
        self.linear_1 = nn.Linear(input_dim, output_dim)

        # Second branch (time embedding)
        self.silu_t = nn.SiLU()
        self.linear_t = nn.Linear(time_embedding_dim, output_dim) # Project time embedding to output_dim

        # Combined path
        self.groupnorm_2 = nn.GroupNorm(num_groups=1, num_channels=output_dim)
        self.silu_2 = nn.SiLU()
        self.dropout = nn.Dropout(0.1)  # Example dropout rate
        self.linear_2 = nn.Linear(output_dim, output_dim) # Linear_0: zero initialization for residual branch


        # Residual connection if input_dim == output_dim, otherwise a projection
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

        # Special initialization
        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, x, t_emb):
        # Store original input for residual connection
        h = x

        # First main path
        x = self.groupnorm_1(x)
        x = self.silu_1(x)
        x = self.linear_1(x)

        # Add time embedding
        t_emb = self.silu_t(t_emb)
        x = x + self.linear_t(t_emb)

        # Second main path
        x = self.groupnorm_2(x)
        x = self.silu_2(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        # Residual connection
        return self.residual_proj(h) + x


class ResMLP(nn.Module):
    def __init__(self, T, data_dim, time_embedding_dim=128, num_blocks=4, hidden_dim=1024):
        super().__init__()
        self.data_dim = data_dim
        self.time_embedding_dim = time_embedding_dim

        # Time embedding
        self.time_embedding_layer = nn.Sequential(
            TimeEmbedding(T, time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim) # As shown in the diagram
        )

        # Input transformation for x
        self.initial_linear_x = nn.Linear(data_dim, hidden_dim) # Initial linear layer for x

        # Sequence of ResMLP blocks
        self.upper_blocks = nn.ModuleList([
            ResBlock(hidden_dim, hidden_dim, time_embedding_dim)
            for _ in range(num_blocks)
        ])

        self.middle_block = ResBlock(hidden_dim, hidden_dim, time_embedding_dim)

        self.lower_blocks = nn.ModuleList([
            ResBlock(hidden_dim, hidden_dim, time_embedding_dim)
            for _ in range(num_blocks)
        ])

        # Output transformation
        self.final_block = ResBlock(hidden_dim, hidden_dim, time_embedding_dim)
        self.final_groupnorm = nn.GroupNorm(num_groups=1, num_channels=hidden_dim)
        self.final_silu = nn.SiLU()
        self.final_linear_zero = nn.Linear(hidden_dim, data_dim) # Linear_0

        nn.init.zeros_(self.final_linear_zero.weight)
        nn.init.zeros_(self.final_linear_zero.bias)


    def forward(self, x, t):
        # 1. Process time embedding
        t_emb = self.time_embedding_layer(t)

        # 2. Process input x
        h = self.initial_linear_x(x)

        hs = [h]

        # 3. Pass through ResMLP blocks
        for block in self.upper_blocks:
            h = block(h, t_emb)
            hs.append(h)

        h = self.middle_block(h, t_emb) + hs.pop()

        for block in self.lower_blocks:
            h = block(h, t_emb) + hs.pop()

        assert len(hs) == 0

        # 4. Final output layers
        h = self.final_block(h, t_emb)
        h = self.final_groupnorm(h)
        h = self.final_silu(h)
        out = self.final_linear_zero(h) # Apply the final Linear_0 as per diagram

        return out


if __name__ == '__main__':
    batch_size = 8
    model = ResMLP(T=1000, data_dim=256, time_embedding_dim=128, num_blocks=4)
    x = torch.randn(batch_size, 256)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)


