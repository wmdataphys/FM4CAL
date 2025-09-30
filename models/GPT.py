import math
import numpy as np
import pkbar

from models.MoE import MoE

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# Prefer BF16 if your GPU supports it; else FP16 is fine
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class ResNetBlock(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.linear1 = nn.Linear(hidden_units, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.activation = nn.ReLU()

    def forward(self, x):
        inputs = x
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x) + inputs)
        return x

    def resnet_subnet(c_in, c_out):
        layers = [nn.Linear(c_in,hidden_units)]

        # Stack residual blocks
        for _ in range(num_blocks):
            layers.append(ResNetBlock(hidden_units))

        layers += [nn.Linear(hidden_units, c_out)]
        return nn.Sequential(*layers)


class TimeRegression(nn.Module):
    def __init__(self, num_blocks, hidden_units, embed_dim):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_units = hidden_units
        self.embed_dim = embed_dim

        layers = [nn.Linear(self.embed_dim, self.hidden_units)]

        for _ in range(self.num_blocks):
            layers.append(ResNetBlock(self.hidden_units))

        layers += [nn.Linear(self.hidden_units, 1), nn.ReLU()]

        self.nn = nn.Sequential(*layers)

    def forward(self, x, k=None):
        return self.nn(x)


class FF(nn.Module):
    def __init__(self, embed_dim, mlp_scale: int = 2,
                 drop_rate: float = 0.0):
        super().__init__()
        self.nn = nn.Sequential(*[nn.Linear(embed_dim, embed_dim * mlp_scale),
                                  nn.GELU(),
                                  nn.Linear(embed_dim * mlp_scale, embed_dim),
                                  nn.Dropout(drop_rate)])

    def forward(self, x):
        return self.nn(x)


class CrossAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 seq_len=27002,
                 dropout=0.2,
                 device="cuda",
                 qk_norm=True):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = embed_dim // num_heads
        self.d_k = self.head_dim ** -0.5
        self.device = device
        self.qk_norm = qk_norm
        self.g_scale = nn.Parameter(torch.tensor(1.0 / self.d_k, dtype=torch.float, device=self.device), requires_grad=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,
                e_embed,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=False,
                past_kv=None):
        """
        x:         [B, T_new, E]  (space stream newest token, T_new=1 at decode)
        e_embed:   [B, T_new, E]  (time stream newest token, T_new=1 at decode)
        past_kv:   Optional[Tuple[Kpast, Vpast]] over x
        Returns: (attn_out, (K_all, V_all))
        """
        batch_size, seq_len, embed_dim = x.shape

        q, k, v = self.q_proj(e_embed), self.k_proj(x), self.v_proj(x)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)  # [B,H,1,D]
        q = q.transpose(1, 2)  # [B,H,1,D]
        v = v.transpose(1, 2)  # [B,H,1,D]

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        # normalize QK matrices with L2 norm over head dim, learnable scale clamped to d_k 
        if self.qk_norm:
            k = F.normalize(k, p=2, dim=-1)
            q = F.normalize(q, p=2, dim=-1)
            attn_scores = self.g_scale * q @ k.transpose(2, 3)

        else:
            attn_scores = self.d_k * q @ k.transpose(2, 3)

        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask, -torch.inf)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]
            attn_scores.masked_fill_(key_padding_mask, -torch.inf)

        attn_scores = F.softmax(attn_scores, dim=-1)  # consider forcing to float32
        attn_scores = self.dropout(attn_scores)

        attn_output = (attn_scores @ v).transpose(1, 2)

        attn_output = attn_output.contiguous().view(batch_size, seq_len, embed_dim)

        if need_weights:
            return attn_output, attn_scores
        else:
            return (attn_output, (k, v))


class CATransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_scale: int = 2,
                 drop_rate: float = 0.2,
                 num_experts: int = 4,
                 num_classes: int = 2,
                 use_MoE: bool = False,
                 device='cuda'):
        super().__init__()
        self.use_MoE = use_MoE
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device
        self.mlp_scale = mlp_scale
        self.drop_rate = drop_rate
        self.xN = nn.LayerNorm(self.embed_dim)
        self.eN = nn.LayerNorm(self.embed_dim)
        self.attn = CrossAttention(self.embed_dim, self.num_heads, dropout=self.drop_rate, device=self.device)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.LN2 = nn.LayerNorm(self.embed_dim)
        if self.use_MoE:
            self.num_experts = num_experts
            self.num_classes = num_classes
            print("CA: ") 
            print("Number of experts: ", self.num_experts)
            print("Num Classes: ", self.num_classes)
            self.FF = MoE(self.embed_dim, mlp_scale=self.mlp_scale, num_experts=self.num_experts, num_classes=self.num_classes)
        else:
            self.FF = FF(self.embed_dim, mlp_scale=self.mlp_scale)

    def generate_mask(self, seq_len):
        return torch.triu(torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool), diagonal=1)

    def forward(self,
                x,
                e_embed,
                class_label,
                padding_mask=None,
                need_weights=False,
                classification=False,
                past_kv=None):
        B, N_t, t_dim = x.shape

        x_norm = self.xN(x)
        e_norm = self.eN(e_embed)
        load_balance = torch.tensor([0.0], dtype=torch.float32, device=x.device)  # place holder for non MoE model return

        # Not used in this model
        # if not classification:
        #     
        #     attn, attn_weights = self.attn(x_norm, e_norm, attn_mask=mask_, key_padding_mask=padding_mask, need_weights=False)
        # else:
        #     attn, attn_weights = self.attn(x_norm, e_norm, key_padding_mask=paddig_mask, need_weights=False)

        mask_ = self.generate_mask(N_t)

        attn_out, kv_ca = self.attn(x_norm,
                                    e_norm,
                                    key_padding_mask=padding_mask,
                                    attn_mask=mask_,
                                    past_kv=past_kv)
        attn_out = self.c_proj(attn_out)
        x = x + attn_out

        if self.use_MoE:
            res, load_balance = self.FF(self.LN2(x), class_label, padding_mask=padding_mask)
            x = x + res
        else:
            x = x + self.FF(self.LN2(x))
            load_balance = torch.zeros((), dtype=torch.float32, device=x.device)

        return x, kv_ca, load_balance


class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, seq_len=250, dropout=0.2, device='cuda', qk_norm=True):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.seq_length = seq_len
        self.head_dim = embed_dim // num_heads
        self.d_k = self.head_dim ** -0.5
        self.device = device
        self.qk_norm = qk_norm
        self.g_scale = nn.Parameter(torch.tensor(1.0 / self.d_k, dtype=torch.float, device=self.device), requires_grad=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,
                attn_mask=None,
                key_padding_mask=None,
                need_weights=False,
                past_kv=None):
        """
        x: [B, T_new, E], where at decode T_new = 1
        past_kv: Optional[Tuple[Kpast, Vpast]], both [B, H, T_past, D]
        Returns: (attn_out, (K_all, V_all))
        """
        batch_size, T_new, embed_dim = x.shape

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        k = k.view(batch_size, T_new, self.num_heads, self.head_dim)
        v = v.view(batch_size, T_new, self.num_heads, self.head_dim)
        q = q.view(batch_size, T_new, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)   # [B,H,T_new,D]
        q = q.transpose(1, 2)   # [B,H,T_new,D]
        v = v.transpose(1, 2)   # [B,H,T_new,D]

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)  # concat on seq dim
            v = torch.cat([past_kv[1], v], dim=2)

        # normalize QK matrices with L2 norm over head dim, learnable scale init to d_k
        if self.qk_norm:
            k = F.normalize(k, p=2, dim=-1)
            q = F.normalize(q, p=2, dim=-1)
            attn_scores = self.g_scale * q @ k.transpose(2, 3)

        else:
            attn_scores = self.d_k * q @ k.transpose(2, 3)

        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask, -torch.inf)

        if key_padding_mask is not None:
            # key_padding_mask: [B, T_total]
            key_padding_mask = key_padding_mask[:, None, None, :]
            attn_scores.masked_fill_(key_padding_mask, -torch.inf)

        attn_scores = F.softmax(attn_scores, dim=-1)  # consider forcing to float32
        attn_scores = self.dropout(attn_scores)

        attn_output = (attn_scores @ v).transpose(1, 2)

        attn_output = attn_output.contiguous().view(batch_size, T_new, embed_dim)

        if need_weights:
            return attn_output, attn_scores
        else:
            return (attn_output, (k, v))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_scale: int = 2, drop_rate: float = 0.2,
                 num_experts: int = 4, num_classes: int = 2, use_MoE: bool = False, device='cuda'):
        super().__init__()
        self.use_MoE = use_MoE
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device
        self.mlp_scale = mlp_scale
        self.drop_rate = drop_rate
        self.LN1 = nn.LayerNorm(self.embed_dim)
        self.attn = MHSA(self.embed_dim, self.num_heads, dropout=self.drop_rate, device=self.device)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.LN2 = nn.LayerNorm(self.embed_dim)

        if self.use_MoE:
            self.num_experts = num_experts
            self.num_classes = num_classes
            print("MHSA: ") 
            print("Number of experts: ", self.num_experts)
            print("Num Classes: ", self.num_classes)
            self.FF = MoE(self.embed_dim, mlp_scale=self.mlp_scale, num_experts=self.num_experts, num_classes=self.num_classes)
        else:
            self.FF = FF(self.embed_dim, mlp_scale=self.mlp_scale)

    def generate_mask(self, seq_len):
        return torch.triu(torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool), diagonal=1)

    def forward(self,
                x,
                class_label,
                padding_mask=None,
                need_weights=False,
                classification=False,
                past_kv=None):
        B, N_t, t_dim = x.shape
        x_norm = self.LN1(x)
        load_balance = torch.tensor([0.0], dtype=torch.float32, device=x.device)  # place holder for non MoE model return

        # Not used in this model
        # if not classification:
        #     mask_ = self.generate_mask(N_t)
        #     attn, attn_weights = self.attn(x_norm, attn_mask=mask_, key_padding_mask=padding_mask, need_weights=need_weights)
        # else:
        #     attn, attn_weights = self.attn(x_norm, key_padding_mask=padding_mask, need_weights=need_weights)

        mask_ = self.generate_mask(N_t)
        attn_out, kv_mhsa = self.attn(
                                x_norm,
                                key_padding_mask=padding_mask,
                                need_weights=need_weights,
                                attn_mask=mask_,
                                past_kv=past_kv)
        attn_out = self.c_proj(attn_out)
        x = x + attn_out

        if self.use_MoE:
            res, _ = self.FF(self.LN2(x), class_label, padding_mask=padding_mask)
            x = x + res
        else:
            x = x + self.FF(self.LN2(x))
            load_balance = torch.zeros((), dtype=torch.float32, device=x.device)

        return x, kv_mhsa, load_balance


class ECAL_GPT(nn.Module):
    def __init__(self,
                 vocab_size,
                 seq_len,
                embed_dim,
                attn_heads=[2, 4, 2],
                num_blocks=2,
                hidden_units=128,
                digitize_energy=True,
                mlp_scale: int = 2,
                energy_vocab: int = 6234,
                space_vocab=27003,
                drop_rates=[0.0, 0.0, 0.0],
                detokenize_func=None,
                classification=False,
                sequence_level=False,
                use_MoE=False, num_experts: int = 4, num_classes: int = 2,
                device='cuda'):
        super().__init__()

        self.use_MoE = use_MoE
        self.classification = classification
        self.sequence_level = sequence_level
        assert not (self.classification and self.use_MoE), "MoE must be off in classification mode"
        assert not (self.sequence_level and self.use_MoE), "MoE must be off in sequence-level mode"
        self.num_experts = num_experts
        self.num_classes = num_classes
        if self.use_MoE:
            print("Using Mixture of Experts.")
        else:
            print("Using traditional FFN.")

        self.digitize_energy = digitize_energy
        self.detokenize_func = detokenize_func
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.energy_pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.initial_energy_embedding = nn.Linear(1, embed_dim)
        self.device = device
        # Can refactor this - fine for now
        layers_ = [CATransformerBlock(embed_dim,
                                      attn_heads[0],
                                      mlp_scale,
                                      drop_rate=drop_rates[0],
                                      use_MoE=self.use_MoE,
                                      num_experts=self.num_experts,
                                      num_classes=self.num_classes,
                                      device=self.device)]
        layers_ += [TransformerBlock(embed_dim,
                                     attn_heads[i],
                                     mlp_scale,
                                     drop_rate=drop_rates[i],
                                     use_MoE=self.use_MoE,
                                     num_experts=self.num_experts,
                                     num_classes=self.num_classes,
                                     device=self.device) for i in range(1, len(attn_heads))]
        self.layers = nn.ModuleList(layers_)
        self.LN = nn.LayerNorm(embed_dim)

        if not self.classification and not self.sequence_level:
            if self.digitize_energy:  # Multiclass
                self.energy_embedding = nn.Embedding(energy_vocab, embed_dim)
                self.energy_head = nn.Linear(embed_dim, energy_vocab)
            else:  # Regression
                self.energy_embedding = nn.Linear(1, embed_dim)
                self.energy_head = TimeRegression(num_blocks, hidden_units, embed_dim)

            self.logits_head = nn.Linear(embed_dim, vocab_size)

        elif self.classification and not self.sequence_level:
            if self.digitize_energy:  # Time resolution based tokenization
                self.energy_embedding = nn.Embedding(energy_vocab, embed_dim)
            else:  # Fully continuous
                self.energy_embedding = nn.Linear(1, embed_dim)

            self.classification_head = nn.Linear(embed_dim, 1)
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        else:
            if self.digitize_energy:  # Time resolution based tokenization
                self.energy_embedding = nn.Embedding(energy_vocab, embed_dim)
            else:  # Fully continuous
                self.energy_embedding = nn.Linear(1, embed_dim)

            self.sequence_head = nn.Linear(embed_dim, 1)

        self.SOS_token = 0
        self.EOS_token = space_vocab - 2  # 6232
        self.pad_token = space_vocab - 1  # 6233
        self.EOS_energy_token = energy_vocab - 2  # 27001
        self.energy_pad_token = energy_vocab - 1  # 27002

    def forward(self, x, e, initial_energy, class_label=None, padding_mask=None):
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        if not self.digitize_energy:
            e = e.reshape(-1, 1)  # [batch_size * seq_len, 1]
            e_embed_flat = self.energy_embedding(e)
            e_embed = e_embed_flat.view(batch_size, seq_len, e_embed_flat.shape[-1])  # [batch_size, seq_len, embed_dim]
            e_embed = e_embed + self.energy_pos_embedding(pos)
        else:
            e_embed = self.energy_embedding(e) + self.energy_pos_embedding(pos)

        # Ensure initial_energy has shape (B, 1) before embedding
        if initial_energy.dim() == 1:
            initial_energy = initial_energy.unsqueeze(1)        # (B, 1)
        elif initial_energy.dim() == 2 and initial_energy.size(1) == 1:
            pass  # already (B, 1)
        else:
            initial_energy = initial_energy.view(batch_size, 1)  # force (B, 1)

        # Embed to (B, embed_dim) then add a time-step dimension -> (B, 1, embed_dim)
        initial_energy_embed = self.initial_energy_embedding(initial_energy)  # (B, E)
        initial_energy_embed = initial_energy_embed.unsqueeze(1)              # (B, 1, E)

        x = self.token_embedding(x) + self.pos_embedding(pos)
        e_embed = torch.cat((initial_energy_embed, e_embed), dim=1)  # Make sure to concat initial energy here
        x = torch.cat((initial_energy_embed, x), dim=1)

        # Instead of adding time and position embeddings, combine through Cross attention
        # Query from time space, given space (key,value)
        if padding_mask is not None:
            energy_mask = torch.zeros(batch_size, initial_energy.shape[-1], dtype=torch.bool, device=x.device)  # No masking for kinematic tokens
            padding_mask = torch.cat((energy_mask, padding_mask), dim=1)

        if self.classification and not self.sequence_level:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)  # no mask for cls token
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)
            x = torch.cat((cls_tokens, x), dim=1)
            e_embed = torch.cat((cls_tokens, e_embed), dim=1)

        if self.training:
            load_balance = torch.zeros((), dtype=torch.float32, device=x.device)
            for layer in self.layers:
                if layer.__class__.__name__ == "CATransformerBlock":
                    x, _kv, load = layer(x, e_embed, class_label, padding_mask=padding_mask, classification=self.classification)
                else:
                    x, _kv, load = layer(x, class_label, padding_mask=padding_mask, classification=self.classification)
                load_balance += load

        else:
            for layer in self.layers:
                if layer.__class__.__name__ == "CATransformerBlock":
                    x, _kv, _lb = layer(x, e_embed, class_label, padding_mask=padding_mask, classification=self.classification)
                else:
                    x, _kv, _lb = layer(x, class_label, padding_mask=padding_mask, classification=self.classification)

        x = self.LN(x)

        if not self.classification and not self.sequence_level:  # Generations - next hit prediction - sequence level will also be false then
            if not self.digitize_energy:
                e_out = self.energy_head(x).squeeze(-1)  # direct regression of time
            else:
                e_out = self.energy_head(x)  # logits over time

            pixel = self.logits_head(x)

            if self.training:
                return pixel, e_out, load_balance
            return pixel, e_out

        elif self.classification and not self.sequence_level:
            return self.classification_head(x[:, 0]).squeeze(-1)
        else:
            return self.sequence_head(x).squeeze(-1)

    def forward_decode_step(self, x_t, e_t, class_label=None,
                            padding_mask=None, past_kvs=None):
        """
        Args:
            x_t: [B,1,E]  newest *space* token embedding (token + pos)
            e_t: [B,1,E]  newest *time*  token embedding (energy + pos)
            past_kvs: list matching self.layers. Each entry is either:
                    - for CA layer: Tuple(K,V) over x-stream
                    - for MHSA layers: Tuple(K,V)
                    or None
        Returns:
            h_t: [B,1,E]  newest hidden
            new_past_kvs: list of updated caches
        """
        if past_kvs is None:
            past_kvs = [None] * len(self.layers)

        x = x_t
        new_past = []
        for layer, pkv in zip(self.layers, past_kvs):
            if isinstance(layer, CATransformerBlock):
                x, kv, _lb = layer(x, e_t, class_label, padding_mask=padding_mask, classification=self.classification, past_kv=pkv)
            else:
                x, kv, _lb = layer(x, class_label, padding_mask=padding_mask, classification=self.classification, past_kv=pkv)
            new_past.append(kv)

        x = self.LN(x)
        return x, new_past

    def __topK(self, logits, topK=50):
        topk_logits, topk_indices = torch.topk(logits, k=topK, dim=-1)
        probs = torch.softmax(topk_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        idx_next = topk_indices.gather(-1, sampled)
        return idx_next

    def __min_p(self, logits, min_p=0.05, min_tokens_to_keep=50, return_logits=False):
        assert 0 <= min_p <= 1, "min_p must be between 0 and 1"

        probs = torch.softmax(logits, dim=-1)
        p_max = torch.max(probs, dim=-1, keepdim=True).values
        p_scaled = min_p * p_max
        min_p_mask = probs < p_scaled

        sorted_indices = torch.argsort(logits, descending=True, dim=-1)
        sorted_indices_to_remove = min_p_mask.gather(-1, sorted_indices)
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False

        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)

        min_p_logits = logits.masked_fill(indices_to_remove, float('-inf'))
        min_p_probs = torch.softmax(min_p_logits, dim=-1)

        sample_token = torch.multinomial(min_p_probs, num_samples=1)

        if return_logits:
            return sample_token, min_p_logits
        return sample_token

    def __nucleus(self, logits, p=0.9):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum_probs = torch.cumsum(probs, dim=-1)

        mask = cumsum_probs > p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False

        sorted_logits[mask] = float('-inf')
        filtered_logits = torch.gather(sorted_logits, -1, torch.argsort(sorted_indices, dim=-1))

        probs = torch.softmax(filtered_logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1) 

        return idx_next

    def __linear_dynamic_temp(self, step, max_length, max_temp=1.0, min_temp=0.95):
        return max(max_temp - step / max_length, min_temp)

    def __exp_dynamic_temp(self, step, max_length, max_temp=1.05, min_temp=0.95):
        alpha = (max_temp - min_temp)
        decay_rate = -math.log(1e-2) / max_length  
        temperature = min_temp + alpha * math.exp(-decay_rate * step)
        return temperature

    @torch.inference_mode()
    def generate(self, initial_energy, class_label=None, max_seq_len=250,
                context_len=None, temperature: float = 1.0, method="Default",
                topK=100, nucleus_p=0.98, dynamic_temp=False):

        device = self.device
        B = initial_energy.shape[0]

        # state we keep for sampling logic only
        idx = torch.zeros(B, 1, device=device, dtype=torch.long)
        e = torch.zeros(B, 1, device=device, dtype=torch.long) if self.digitize_energy else torch.zeros(B, 1, device=device).float()

        # initial-energy token embedding (t = 0)
        if initial_energy.dim() == 1:
            initial_energy = initial_energy.unsqueeze(1)
        init_e_embed = self.initial_energy_embedding(initial_energy).unsqueeze(1)  # [B,1,E]

        # caches per layer
        past_kvs = [None] * len(self.layers)

        is_done = torch.zeros(B, dtype=torch.bool, device=device)

        # position counter (we insert initial-energy at t=0; next is t=1)
        t = 0

        with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ):
            for step in range(max_seq_len):
                # build current-step embeddings
                if step == 0:
                    # for t=0, we query with initial-energy on both streams
                    x_t = init_e_embed  # space stream includes the init energy first (per your design)
                    e_t = init_e_embed
                else:
                    # position index for new token
                    pos_idx = torch.full((B, 1), t, device=device, dtype=torch.long)
                    # token stream
                    x_t = self.token_embedding(idx[:, -1:]) + self.pos_embedding(pos_idx)
                    # time/energy stream
                    if self.digitize_energy:
                        e_t = self.energy_embedding(e[:, -1:]) + self.energy_pos_embedding(pos_idx)
                    else:
                        e_t = self.energy_embedding(e[:, -1:].reshape(-1,1)).view(B, 1,-1) + self.energy_pos_embedding(pos_idx)

                # one decode step through the stack w/ caches
                h_t, past_kvs = self.forward_decode_step(x_t, e_t, class_label=class_label, padding_mask=None, past_kvs=past_kvs)

                # project to logits
                if not self.classification and not self.sequence_level:
                    pixel_logits = self.logits_head(h_t)[:, -1, :] / temperature
                    if self.digitize_energy:
                        time_logits = self.energy_head(h_t)[:, -1, :] / temperature
                    else:
                        time_val = self.energy_head(h_t)[:, -1]  # regression

                    # sample pixel token
                    if method == "Default":
                        probs = F.softmax(pixel_logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                    elif method == "TopK":
                        idx_next = self.__topK(pixel_logits, topK)
                    elif method == "Nucleus":
                        idx_next = self.__nucleus(pixel_logits, nucleus_p)
                    elif method == "Greedy":
                        idx_next = torch.argmax(pixel_logits, dim=-1, keepdim=True)
                    elif method == "Min_p":
                        idx_next = self.__min_p(pixel_logits)

                    # sample time token/value
                    if self.digitize_energy:
                        probs_t = F.softmax(time_logits, dim=-1, dtype=torch.float32)
                        e_next = torch.multinomial(probs_t, num_samples=1) if method=="Default" else \
                                (self.__topK(time_logits, topK) if method=="TopK" else
                                self.__nucleus(time_logits, nucleus_p) if method=="Nucleus" else
                                torch.argmax(time_logits, dim=-1, keepdim=True) if method=="Greedy" else
                                self.__min_p(time_logits))
                    else:
                        e_next = time_val.unsqueeze(1)

                    # EOS handling
                    is_done |= (e_next.squeeze(1) == self.EOS_energy_token) | (idx_next.squeeze(1) == self.EOS_token)
                    idx_next[is_done] = self.EOS_token
                    if self.digitize_energy:
                        e_next[is_done] = self.EOS_energy_token

                    # append to sequences (for next-step embedding only)
                    idx = torch.cat([idx, idx_next], dim=1)
                    e = torch.cat([e, e_next], dim=1)

                    t += 1
                    if torch.all(is_done):
                        break
                else:
                    raise NotImplementedError("KV-cached generation shown for the autoregressive (non-classification, non-sequence-level) path.")
        return idx, e

    @torch.no_grad()
    def generate_PDF(self, kinematics, unscaled_k, PID=None, numPhotons=2e5, max_seq_len: int = 250,
                 context_len=None, temperature: float = 1.05, method="Nucleus", topK=100,
                 nucleus_p=0.995, dynamic_temp=False, add_dark_noise=False):

        assert kinematics is not None

        batch_size = kinematics.shape[0]
        kbar = pkbar.Kbar(target=numPhotons, width=20, always_stateful=False)

        torch.cuda.empty_cache()
        tracks = []
        n_total = 0

        if PID == "Pion" and self.use_MoE:
            class_label = torch.zeros((batch_size,), dtype=torch.float32, device=kinematics.device)
        elif PID == "Kaon" and self.use_MoE:
            class_label = torch.ones((batch_size,), dtype=torch.float32, device=kinematics.device)
        else:
            class_label = None

        while n_total < numPhotons:

            with torch.no_grad():
                track = self.generate(kinematics, unscaled_k, class_label=class_label, method=method, temperature=temperature,
                                      topK=topK, nucleus_p=nucleus_p, dynamic_temp=dynamic_temp, add_dark_noise=add_dark_noise)

            tracks += track
            n_generated = self.__count_photons(track)
            n_total += n_generated

            kbar.add(n_generated)

        torch.cuda.empty_cache()


        xs, ys, times = [], [], []

        for track_ in tracks:
            xs.append(track_['x'])
            ys.append(track_['y'])
            times.append(track_['leadTime'])

        xs = np.concatenate(xs)[:numPhotons]
        ys = np.concatenate(ys)[:numPhotons]
        times = np.concatenate(times)[:numPhotons]
        return {"x": xs, "y": ys, "leadTime": times}

    def __count_photons(self, tracks):
        counter = 0
        for track in tracks:
            counter += track["NHits"]

        return counter
