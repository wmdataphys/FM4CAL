import math
import torch 
import torch.nn as nn

def decompose_weights(weight_matrix, r):
    # weight_matrix: [D, D]
    u, s, v = torch.linalg.svd(weight_matrix, full_matrices=False)
    s = s.diag()
    u_r = u[:, :r]  
    s_r = s[:r,:r]
    v_r = v[:,:r]
    A = u_r @ torch.sqrt(s_r) # [D, r]
    B = torch.sqrt(s_r) @ v_r.T # [r, D]
    return A,B

def decompose_vocab_weights(weight_matrix, r):
    # weight_matrix: [V, D]
    u, s, v = torch.linalg.svd(weight_matrix, full_matrices=False)
    s = s.diag()
    u_r = u[:, :r]          # [V, r]
    s_r = s[:r,:r]             # [r,r]
    v_r = v[:r, :]          # [r, D]
    sqrt_s = torch.sqrt(s_r)
    A = sqrt_s @  v_r        # [r, D]
    B = u_r @ sqrt_s        # [V, r]
    return A,B

class LoRA(nn.Module):
    def __init__(self, embed_dim, lora_r, weights=None, alpha=1.0, drop_rate=0.0, mlp_scale=2, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.lora_r = lora_r
        self.scale = alpha / math.sqrt(lora_r)
        print(f"Using rank stabilization: scale = {self.scale}")
        print(f"LoRA: embed_dim={embed_dim}, rank={lora_r}, alpha={alpha}, drop_rate={drop_rate}, mlp_scale={mlp_scale}")
        self.device = device 
        if weights is None:
            # LoRA for Q
            self.lora_A_Q = nn.Parameter(torch.randn(embed_dim, lora_r) * 0.01)
            self.lora_B_Q = nn.Parameter(torch.zeros(lora_r, embed_dim))

            # LoRA for K
            self.lora_A_K = nn.Parameter(torch.randn(embed_dim, lora_r) * 0.01)
            self.lora_B_K = nn.Parameter(torch.zeros(lora_r, embed_dim))

            # LoRA for V
            self.lora_A_V = nn.Parameter(torch.randn(embed_dim, lora_r) * 0.01)
            self.lora_B_V = nn.Parameter(torch.zeros(lora_r, embed_dim))

            # LoRA for output projection 
            self.lora_A_c_proj = nn.Parameter(torch.randn(embed_dim, lora_r) * 0.01)
            self.lora_B_c_proj = nn.Parameter(torch.zeros(lora_r, embed_dim))
        else:
            # If weights are provided, apply PiSSA decomp
            # I need to further factor this like vocab LoRA - not used at the moment.
            print("Using PiSSA decomposition for LoRA")
            Q, K, V, c_c_proj = weights
            A_Q, B_Q = decompose_weights(Q, lora_r)
            A_K, B_K = decompose_weights(K, lora_r)
            A_V, B_V = decompose_weights(V, lora_r)
            A_proj, B_proj = decompose_weights(c_c_proj, lora_r)
            
            self.lora_A_Q = nn.Parameter(A_Q)
            self.lora_B_Q = nn.Parameter(B_Q)
            self.lora_A_K = nn.Parameter(A_K)
            self.lora_B_K = nn.Parameter(B_K)
            self.lora_A_V = nn.Parameter(A_V)
            self.lora_B_V = nn.Parameter(B_V)
            self.lora_A_c_proj = nn.Parameter(A_proj)
            self.lora_B_c_proj = nn.Parameter(B_proj)

        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else lambda x: x


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
    def __init__(self, vocab_size, embed_dim, lora_r, weight=None, alpha=1.0, drop_rate=0.0, device='cuda',pissa_init=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lora_r = lora_r
        self.scale = alpha / math.sqrt(lora_r)
        self.device = device

        if weight is None:
            weight = torch.randn(vocab_size, embed_dim) * 0.01

        if not pissa_init:
            print("Using standard LoRA initialization for Vocab LoRA")
            self.lora_A_vocab = nn.Parameter(torch.randn(lora_r, embed_dim) * 0.01) # A: [r, D], B: [V, r]
            self.lora_B_vocab = nn.Parameter(torch.zeros(vocab_size, lora_r))
            print("Registering weight buffer for original vocab weight")
            self.register_buffer('residual_weight',weight)
            print(self.residual_weight)
        else:
            # PiSSA init: weight is [V, D]
            print("Using PiSSA decomposition for Vocab LoRA")
            A, B = decompose_vocab_weights(weight, lora_r) # A: [r, D], B: [V, r]
            self.lora_A_vocab = nn.Parameter(A)  
            self.lora_B_vocab = nn.Parameter(B) 
            self.scale = 1.0  # No scaling for PiSSA init
            res = weight - (B @ A) 
            self.register_buffer('residual_weight', res)
            print("Registering weight buffer for residual weight")
            print(self.residual_weight)

        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward(self, x):
        x_d = self.dropout(x)
        h = x_d @ self.lora_A_vocab.T
        delta_logits = (h @ self.lora_B_vocab.T) * self.scale
        base_logits = x_d @ self.residual_weight.T
        return base_logits + delta_logits


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


class LPE_Expansion(nn.Module):
    def __init__(self, embed_dim, seq_len, weight=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        if weight is not None:
            print(f"Initializing LPE with provided {weight.shape} weight matrix.")
            self.pos_embedding.weight[:weight.shape[0]].data.copy_(weight)
            self.old_seq_len = weight.shape[0]
        else:
            print("Initializing LPE with random weights.")
            self.old_seq_len = 0

    def mask_old_LPE_grads(self,grad):
        grad_masked = grad.clone()
        grad_masked[:self.old_seq_len, :] = 0.0
        return grad_masked

    def forward(self, positions):
        pos_embeds = self.pos_embedding(positions)

        if self.training and pos_embeds.requires_grad and self.old_seq_len > 0:
            pos_embeds.register_hook(self.mask_old_LPE_grads)

        return pos_embeds


