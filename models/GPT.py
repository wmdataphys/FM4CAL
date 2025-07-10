import math
import numpy as np
import pkbar

from models.MoE import MoE

import torch
import torch.nn as nn
from torch.nn import functional as F


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

        assert embed_dim & num_heads == 0, "embed_dim is indivisible by num_heads"

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
                need_weights=False):
        batch_size, seq_len, embed_dim = x.shape

        q, k, v = self.q_proj(e_embed), self.k_proj(x), self.v_proj(x)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

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
            return attn_output, None


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

    def forward(self, x, e_embed, class_label, padding_mask=None, need_weights=False, classification=False):
        B, N_t, t_dim = x.shape
        x_norm = self.xN(x)
        e_norm = self.eN(e_embed)
        load_balance = torch.tensor([0.0], dtype=torch.float32, device=x.device)  # place holder for non MoE model return

        if not classification:
            mask_ = self.generate_mask(N_t)
            attn, attn_weights = self.attn(x_norm, e_norm, attn_mask=mask_, key_padding_mask=padding_mask, need_weights=False)
        else:
            attn, attn_weights = self.attn(x_norm, e_norm, key_padding_mask=padding_mask, need_weights=False)

        attn = self.c_proj(attn)
        x = x + attn

        if self.use_MoE:
            res, load_balance = self.FF(self.LN2(x), class_label, padding_mask=padding_mask)
            x = x + res
        else:
            x = x + self.FF(self.LN2(x))
        return x, load_balance


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
        self.g_scale = nn.Parameter(torch.tensor(1.0/self.d_k, dtype=torch.float, device=self.device), requires_grad=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None, need_weights=False):
        batch_size, seq_len, embed_dim = x.shape

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

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
            key_padding_mask = key_padding_mask[:, None, None, :]
            attn_scores.masked_fill_(key_padding_mask, -torch.inf)

        attn_scores = F.softmax(attn_scores, dim=-1)  # consider forcing to float32
        attn_scores = self.dropout(attn_scores)

        attn_output = (attn_scores @ v).transpose(1, 2)

        attn_output = attn_output.contiguous().view(batch_size, seq_len, embed_dim)

        if need_weights:
            return attn_output, attn_scores
        else:
            return attn_output, None


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

    def forward(self, x, class_label, padding_mask=None, need_weights=False, classification=False):
        B, N_t, t_dim = x.shape
        x_norm = self.LN1(x)
        load_balance = torch.tensor([0.0], dtype=torch.float32, device=x.device)  # place holder for non MoE model return

        if not classification:
            mask_ = self.generate_mask(N_t)
            attn, attn_weights = self.attn(x_norm, attn_mask=mask_, key_padding_mask=padding_mask, need_weights=need_weights)
        else:
            attn, attn_weights = self.attn(x_norm, key_padding_mask=padding_mask, need_weights=need_weights)

        attn = self.c_proj(attn)

        x = x + attn

        if self.use_MoE:
            res, load_balance = self.FF(self.LN2(x), class_label, padding_mask=padding_mask)
            x = x + res
        else:
            x = x + self.FF(self.LN2(x))

        return x, load_balance


class ECAL_GPT(nn.Module):
    def __init__(self,
                 vocab_size,
                 seq_len,
                embed_dim, attn_heads=[2, 4, 2],
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

        # Ensure initial_energy is 1D before embedding
        initial_energy = initial_energy.unsqueeze(1)  # [batch_size, 1]

        initial_energy_embed = self.initial_energy_embedding(initial_energy).unsqueeze(1)
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
            load_balance = 0.
            for layer in self.layers:
                if layer.__class__.__name__ == "CATransformerBlock":
                    x, load = layer(x, e_embed, class_label, padding_mask=padding_mask, classification=self.classification)
                else:
                    x, load = layer(x, class_label, padding_mask=padding_mask, classification=self.classification)
                load_balance += load

        else:
            for layer in self.layers:
                if layer.__class__.__name__ == "CATransformerBlock":
                    x, _ = layer(x, e_embed, class_label, padding_mask=padding_mask, classification=self.classification)
                else:
                    x, _ = layer(x, class_label, padding_mask=padding_mask, classification=self.classification)

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

    @torch.no_grad()
    def generate(self, inital_energy, class_label=None, max_seq_len: int = 250, context_len=None,
                 temperature: float = 1.0, method="Default", topK=100, nucleus_p=0.98,
                 dynamic_temp=False):

        assert method in ["Nucleus", "TopK", "Default", "Greedy", "Min_p"]
        batch_size = inital_energy.shape[0]

        # Start tokens
        idx = torch.zeros(batch_size, 1, device=self.device, dtype=torch.long)  # pixel token
        if self.digitize_energy:
            e = torch.zeros(batch_size, 1, device=self.device, dtype=torch.long)  # time token
        else:
            e = torch.zeros(batch_size, 1).to(self.device).float()

        is_done = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for step_ in range(max_seq_len):
            if dynamic_temp:
                temperature = self.__exp_dynamic_temp(step_, max_seq_len)

            if context_len is None:
                idx_cond = idx
                e_cond = e
            else:
                idx_cond = idx[:, -context_len:]
                e_cond = e[:, -context_len:]

            logits, logits_energy = self(idx_cond, e_cond, inital_energy, class_label, padding_mask=None)

            logits = logits[:, -1, :] / temperature

            # ---- Pixel sampling ----
            if method == "Default":
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            elif method == "TopK":
                idx_next = self.__topK(logits, topK)
            elif method == "Nucleus":
                idx_next = self.__nucleus(logits, nucleus_p)
            elif method == "Greedy":
                idx_next = torch.argmax(logits, dim=-1).unsqueeze(1)
            elif method == "Min_p":
                idx_next = self.__min_p(logits, return_logits=False)
            elif method == "Typical":
                idx_next = self.__typical_sampling(logits, return_logits=False)

            # ---- Time sampling ----
            if self.digitize_energy:
                logits_energy = logits_energy[:, -1, :] / temperature
                if method == "Default":
                    probs_time = F.softmax(logits_energy, dim=-1, dtype=torch.float32)
                    e_next = torch.multinomial(probs_time, num_samples=1)
                elif method == "TopK":
                    e_next = self.__topK(logits_energy, topK)
                elif method == "Nucleus":
                    e_next = self.__nucleus(logits_energy, nucleus_p)
                elif method == "Greedy":
                    e_next = torch.argmax(logits_energy, dim=-1).unsqueeze(1)
                elif method == "Min_p":
                    e_next = self.__min_p(logits_energy, return_logits=False)
                elif method == "Typical":
                    e_next = self.__typical_sampling(logits_energy, return_logits=False)
            else:
                e_next = logits_energy[:, -1].unsqueeze(1)

            # pixel, or time
            is_done |= (e_next.squeeze(1) == self.EOS_energy_token) | (idx_next.squeeze(1) == self.EOS_token)

            idx_next[is_done] = self.EOS_token
            e_next[is_done] = self.EOS_energy_token

            idx = torch.cat((idx, idx_next), dim=1)
            e = torch.cat((e, e_next), dim=1)

            if torch.all(is_done):
                break

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
