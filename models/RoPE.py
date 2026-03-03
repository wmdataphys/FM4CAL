import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

class PartialRoPE(nn.Module):
    def __init__(self, embed_dim, num_heads, rope_fraction=0.5, theta=1000):
        super().__init__()
        self.rope_fraction = rope_fraction
        head_dim = embed_dim // num_heads
        rope_dim = int(head_dim * rope_fraction)
        
        self.rope = RotaryEmbedding(
            dim=rope_dim,
            cache_if_possible=True,
            theta=theta
        )
        self.rope_dim = rope_dim
        self.head_dim = head_dim

    def forward(self, q, k, offset=0):
        q_rope = q[..., :self.rope_dim]
        q_no_rope = q[..., self.rope_dim:]
        
        k_rope = k[..., :self.rope_dim]
        k_no_rope = k[..., self.rope_dim:]
        
        q_rope = self.rope.rotate_queries_or_keys(q_rope, offset=offset)
        k_rope = self.rope.rotate_queries_or_keys(k_rope, offset=offset)
        
        q = torch.cat([q_rope, q_no_rope], dim=-1)
        k = torch.cat([k_rope, k_no_rope], dim=-1)
            
        return q, k