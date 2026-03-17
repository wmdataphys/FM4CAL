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

        self._precomputed_freqs = None

    def precompute_freqs(self, max_seq_len, device, dtype=torch.float32):
        seq = torch.arange(max_seq_len, device=device, dtype=dtype)
        # This returns the raw frequency tensor the library uses internally
        self._precomputed_freqs = self.rope.forward(seq)  # [max_seq_len, rope_dim/2] or [max_seq_len, rope_dim]

    def forward(self, q, k, offset=0):
        q_rope = q[..., :self.rope_dim]
        q_no_rope = q[..., self.rope_dim:]
        k_rope = k[..., :self.rope_dim]
        k_no_rope = k[..., self.rope_dim:]

        if self._precomputed_freqs is not None:
            seq_len = q_rope.shape[-2]  # Python int from shape
            if seq_len == 1 and isinstance(offset, torch.Tensor):
                # CUDA graph safe: single-position lookup via index_select
                freqs = self._precomputed_freqs.index_select(0, offset.long().view(1))
            else:
                # Training/prefill: offset is a Python int, slicing is fine
                freqs = self._precomputed_freqs[offset:offset + seq_len]
            from rotary_embedding_torch import apply_rotary_emb
            q_rope = apply_rotary_emb(freqs, q_rope)
            k_rope = apply_rotary_emb(freqs, k_rope)
        else:
            q_rope = self.rope.rotate_queries_or_keys(q_rope, offset=offset)
            k_rope = self.rope.rotate_queries_or_keys(k_rope, offset=offset)

        q = torch.cat([q_rope, q_no_rope], dim=-1)
        k = torch.cat([k_rope, k_no_rope], dim=-1)
        return q, k