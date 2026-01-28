import math
import torch 
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, embed_dim, lora_r, alpha=1.0, drop_rate=0.0, mlp_scale=2, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.lora_r = lora_r
        self.scale = alpha / math.sqrt(lora_r)
        print(f"Using rank stabilization: scale = {self.scale}")
        print(f"LoRA: embed_dim={embed_dim}, rank={lora_r}, alpha={alpha}, drop_rate={drop_rate}, mlp_scale={mlp_scale}")
        self.device = device 

        # LoRA for Q
        self.lora_A_Q = nn.Parameter(torch.randn(embed_dim, lora_r) * 0.01)
        self.lora_B_Q = nn.Parameter(torch.zeros(lora_r, embed_dim))

        # LoRA for K
        self.lora_A_K = nn.Parameter(torch.randn(embed_dim, lora_r) * 0.01)
        self.lora_B_K = nn.Parameter(torch.zeros(lora_r, embed_dim))

        # LoRA for V
        self.lora_A_V = nn.Parameter(torch.randn(embed_dim, lora_r) * 0.01)
        self.lora_B_V = nn.Parameter(torch.zeros(lora_r, embed_dim))

        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else lambda x: x

        # LoRA for output projection 
        self.lora_A_c_proj = nn.Parameter(torch.randn(embed_dim, lora_r) * 0.01)
        self.lora_B_c_proj = nn.Parameter(torch.zeros(lora_r, embed_dim))

    def forward_proj(self, x):
        x_d = self.dropout(x)

        delta_out = (x_d @ self.lora_A_c_proj @ self.lora_B_c_proj) * self.scale

        return delta_out


    def forward(self, x,e_embed=None):
        if e_embed is None:
            x_d = self.dropout(x)
            delta_Q = (x_d @ self.lora_A_Q @ self.lora_B_Q) * self.scale
            delta_K = (x_d @ self.lora_A_K @ self.lora_B_K) * self.scale
            delta_V = (x_d @ self.lora_A_V @ self.lora_B_V) * self.scale

            return delta_Q, delta_K, delta_V

        else:
            x_d,e_d = self.dropout(x),self.dropout(e_embed)
            # Query comes from energy in CA
            delta_Q = (e_d @ self.lora_A_Q @ self.lora_B_Q) * self.scale
            # Key, Value from token embeddings
            delta_K = (x_d @ self.lora_A_K @ self.lora_B_K) * self.scale
            delta_V = (x_d @ self.lora_A_V @ self.lora_B_V) * self.scale
            return delta_Q, delta_K, delta_V

class Embed_LoRA(nn.Module):
    def __init__(self, embed_dim, lora_r, alpha=1.0, drop_rate=0.0, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.lora_r = lora_r
        self.scale = alpha / math.sqrt(lora_r)
        print(f"Using rank stabilization: scale = {self.scale}")
        print(f"Embed LoRA: embed_dim={embed_dim}, rank={lora_r}, alpha={alpha}, drop_rate={drop_rate}")
        self.device = device

        self.lora_A_x = nn.Parameter(torch.randn(embed_dim, lora_r) * 0.01)
        self.lora_B_x = nn.Parameter(torch.zeros(lora_r, embed_dim))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else lambda x: x

    def forward(self, x):
        x_d = self.dropout(x)

        delta_x = (x_d @ self.lora_A_x @ self.lora_B_x) * self.scale


        return delta_x

class Vocab_LoRA(nn.Module):
    def __init__(self, vocab_size, embed_dim, lora_r, alpha=1.0, drop_rate=0.0, device='cuda'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lora_r = lora_r
        self.scale = alpha / math.sqrt(lora_r)
        print(f"Vocab using rank stabilization: scale = {self.scale}")
        print(f"Vocab LoRA: vocab_size={vocab_size}, embed_dim={embed_dim}, rank={lora_r}, alpha={alpha}, drop_rate={drop_rate}")
        self.device = device

        self.lora_A_vocab = nn.Parameter(torch.randn(lora_r, embed_dim) * 0.01) # [r, D]
        self.lora_B_vocab = nn.Parameter(torch.zeros(vocab_size, lora_r)) # [V, r]

        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else lambda x: x

    def forward(self, x):
        x_d = self.dropout(x) # [batch_size, embed_dim]
        h = x_d @ self.lora_A_vocab.T # [batch_size, r]
        delta_x = h @ self.lora_B_vocab.T * self.scale # [batch_size, S, V]

        return delta_x

class ConditionedAdapter(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim=16):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, embed_dim)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x,):
        return x + self.mlp(x)
