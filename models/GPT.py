import math
import numpy as np
import pkbar
import copy

from models.MoE import MoE, Router, Expert
from models.LoRA import LoRA ,Embed_LoRA, Vocab_LoRA, ConditionedAdapter

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import scaled_dot_product_attention as sdpa
import math

torch.backends.cuda.enable_flash_sdp(True)         # use FlashAttention when possible
torch.backends.cuda.enable_mem_efficient_sdp(True) # fallback fused kernel
torch.backends.cuda.enable_math_sdp(False)         # prefer the fast paths

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
# Prefer BF16 if your GPU supports it; else FP16 is fine
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


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

    def forward(self, x, e_embed, attn_mask=None, key_padding_mask=None,
                need_weights=False, past_kv=None, LoRA_module=None):
        """
        x:         [B, T_new, E]
        e_embed:   [B, T_new, E]
        past_kv:   Dict with {'k': cache_k, 'v': cache_v, 'seq_len': int} or None
        Returns: (attn_out, updated_cache_dict)
        """
        batch_size, seq_len, embed_dim = x.shape

        q, k, v = self.q_proj(e_embed), self.k_proj(x), self.v_proj(x)

        # Apply LoRA to Q,K if available
        if LoRA_module is not None:
            IA3_K,IA3_V = LoRA_module.get_IA3_KV()
            delta_Q, delta_K, delta_V = LoRA_module(x,e_embed=e_embed)  # (B, T_new, E)
            q = q + delta_Q
            k = (k + delta_K) * IA3_K
            v = (v + delta_V) * IA3_V

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)  # [B, H, T_new, D]
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Normalize Q and K
        if self.qk_norm:
            k = F.normalize(k, p=2, dim=-1)
            q = F.normalize(q, p=2, dim=-1)

        # Handle KV cache with pre-allocation
        if past_kv is not None:
            cache_k = past_kv['k']
            cache_v = past_kv['v']
            curr_len = past_kv['seq_len']
            
            # Write new K,V into pre-allocated cache
            cache_k[:, :, curr_len:curr_len+seq_len] = k
            cache_v[:, :, curr_len:curr_len+seq_len] = v
            
            # Update sequence length
            new_len = curr_len + seq_len
            
            # Use cache up to current position
            k = cache_k[:, :, :new_len]
            v = cache_v[:, :, :new_len]
            
            # Update cache dict for return
            updated_cache = {
                'k': cache_k,
                'v': cache_v,
                'seq_len': new_len
            }
        else:
            updated_cache = None

        # Compute attention scores
        if self.qk_norm:
            attn_scores = self.g_scale * q @ k.transpose(2, 3)
        else:
            attn_scores = self.d_k * q @ k.transpose(2, 3)

        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask, -torch.inf)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]
            attn_scores.masked_fill_(key_padding_mask, -torch.inf)

        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        attn_output = (attn_scores @ v).transpose(1, 2)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, embed_dim)

        if need_weights:
            return attn_output, attn_scores
        else:
            return (attn_output, updated_cache)


class CATransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_scale: int = 2,
                 drop_rate: float = 0.2,
                 num_experts: int = 4,
                 num_classes: int = 2,
                 use_MoE: bool = False,
                 device='cuda',
                 LoRA_module=None):
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
                material_index,
                padding_mask=None,
                need_weights=False,
                classification=False,
                past_kv=None, LoRA_module=None):
        B, N_t, t_dim = x.shape

        x_norm = self.xN(x)
        e_norm = self.eN(e_embed)
        load_balance = torch.tensor([0.0], dtype=torch.float32, device=x.device)  # place holder for non MoE model return

        need_causal = (past_kv is None) or (past_kv is not None and past_kv["seq_len"] == 0 and N_t > 1)
        mask_ = self.generate_mask(N_t) if need_causal else None


        attn_out, kv_ca = self.attn(x_norm,
                                    e_norm,
                                    key_padding_mask=padding_mask,
                                    attn_mask=mask_,
                                    past_kv=past_kv,
                                    LoRA_module=LoRA_module)

        delta_proj = LoRA_module.forward_proj(attn_out) if LoRA_module is not None else 0.0
        attn_out = self.c_proj(attn_out) + delta_proj

        x = x + attn_out

        if self.use_MoE:
            res, load_balance = self.FF(self.LN2(x), material_index, padding_mask=padding_mask)
            x = x + res
        else:
            x = x + self.FF(self.LN2(x))

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

    def forward(self, x, attn_mask=None, key_padding_mask=None,
                need_weights=False, past_kv=None, LoRA_module=None):
        """
        x: [B, T_new, E]
        past_kv: Dict with {'k': cache_k, 'v': cache_v, 'seq_len': int} or None
        Returns: (attn_out, updated_cache_dict)
        """
        batch_size, T_new, embed_dim = x.shape

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Apply LoRA to Q,K if available
        if LoRA_module is not None:
            IA3_K,IA3_V = LoRA_module.get_IA3_KV()
            delta_Q, delta_K, delta_V = LoRA_module(x)  # (B, T_new, E)
            q = q + delta_Q
            k = (k + delta_K) * IA3_K
            v = (v + delta_V) * IA3_V


        k = k.view(batch_size, T_new, self.num_heads, self.head_dim)
        v = v.view(batch_size, T_new, self.num_heads, self.head_dim)
        q = q.view(batch_size, T_new, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)  # [B, H, T_new, D]
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Normalize Q and K
        if self.qk_norm:
            k = F.normalize(k, p=2, dim=-1)
            q = F.normalize(q, p=2, dim=-1)

        # Handle KV cache with pre-allocation
        if past_kv is not None:
            cache_k = past_kv['k']
            cache_v = past_kv['v']
            curr_len = past_kv['seq_len']
            
            # Write new K,V into pre-allocated cache
            cache_k[:, :, curr_len:curr_len+T_new] = k
            cache_v[:, :, curr_len:curr_len+T_new] = v
            
            # Update sequence length
            new_len = curr_len + T_new
            
            # Use cache up to current position
            k = cache_k[:, :, :new_len]
            v = cache_v[:, :, :new_len]
            
            # Update cache dict for return
            updated_cache = {
                'k': cache_k,
                'v': cache_v,
                'seq_len': new_len
            }
        else:
            updated_cache = None

        # Compute attention scores
        if self.qk_norm:
            attn_scores = self.g_scale * q @ k.transpose(2, 3)
        else:
            attn_scores = self.d_k * q @ k.transpose(2, 3)

        if attn_mask is not None:
            attn_scores.masked_fill_(attn_mask, -torch.inf)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask[:, None, None, :]
            attn_scores.masked_fill_(key_padding_mask, -torch.inf)

        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        attn_output = (attn_scores @ v).transpose(1, 2)
        attn_output = attn_output.contiguous().view(batch_size, T_new, embed_dim)

        if need_weights:
            return attn_output, attn_scores
        else:
            return (attn_output, updated_cache)


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
                material_index,
                padding_mask=None,
                need_weights=False,
                classification=False,
                past_kv=None,
                LoRA_module=None):
        B, N_t, t_dim = x.shape
        x_norm = self.LN1(x)
        load_balance = torch.tensor([0.0], dtype=torch.float32, device=x.device)  # place holder for non MoE model return

        need_causal = (past_kv is None) or (past_kv is not None and past_kv["seq_len"] == 0 and N_t > 1)
        mask_ = self.generate_mask(N_t) if need_causal else None


        attn_out, kv_mhsa = self.attn(
                                x_norm,
                                key_padding_mask=padding_mask,
                                need_weights=need_weights,
                                attn_mask=mask_,
                                past_kv=past_kv, LoRA_module=LoRA_module)

        delta_proj = LoRA_module.forward_proj(attn_out) if LoRA_module is not None else 0.0
        attn_out = self.c_proj(attn_out) + delta_proj
        x = x + attn_out

        if self.use_MoE:
            res, load_balance = self.FF(self.LN2(x), material_index, padding_mask=padding_mask)
            x = x + res
        else:
            x = x + self.FF(self.LN2(x))

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
                use_MoE=False, num_experts: int = 2, material_list: list  = ["G4_W_gamma", "G4_Ta_gamma"],
                particle_list: list = ["gamma"], base_model_type: str = "gamma",
                device='cuda',
                grid_shape=None,
                LoRA_alpha=64,
                LoRA_r=32,
                T_ref=1000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.attn_heads = attn_heads
        self.num_blocks = num_blocks  
        self.mlp_scale = mlp_scale  
        self.drop_rates = drop_rates
        self.use_MoE = use_MoE
        self.classification = classification
        self.sequence_level = sequence_level
        self.base_model_type = base_model_type
        self.LoRA_alpha = LoRA_alpha
        self.LoRA_r = LoRA_r

        self.material_list = material_list
        self.num_experts = num_experts
        self.num_classes = len(self.material_list)
        self.particle_list = particle_list
        
        if self.use_MoE:
            print(f"Using Mixture of Experts for materials: {self.material_list}.")
            print("Fine tuning will expand experts and/or LoRA modules for particles.")
        else:
            print("Using traditional FFN.")

        self.digitize_energy = digitize_energy
        self.detokenize_func = detokenize_func
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.energy_pos_embedding = nn.Embedding(seq_len, embed_dim)
        self.initial_energy_embedding = nn.Linear(1, embed_dim)
        self.LN = nn.LayerNorm(embed_dim)
        self.energy_embedding = nn.Embedding(energy_vocab, embed_dim)
        self.energy_head = nn.Linear(embed_dim, energy_vocab)
        self.logits_head = nn.Linear(embed_dim, vocab_size)
        print("Not using material embeddings.")
        #self.material_embedding = nn.Embedding(self.num_classes, embed_dim)
        self.device = device
        self.space_vocab = space_vocab
        self.energy_vocab = energy_vocab

        # Add in 3D positional embeddings if provided
        self.grid_shape = grid_shape
        if grid_shape is not None:
            Nz, Ny, Nx = grid_shape
            self.Nz, self.Ny, self.Nx = Nz, Ny, Nx
            num_cells = Nz * Ny * Nx

            self.x_embedding = nn.Embedding(Nx, embed_dim)
            self.y_embedding = nn.Embedding(Ny, embed_dim)
            self.z_embedding = nn.Embedding(Nz, embed_dim)

            # Precompute mapping tokens -> (z,y,x)
            # token ids occupy [1, num_cells], 0 is SOS, num_cells+1 is EOS, num_cells+2 is PAD
            flat = torch.arange(num_cells, dtype=torch.long)
            z = flat // (Ny * Nx)
            rem = flat % (Ny * Nx)
            y = rem // Nx
            x = rem % Nx

            # Allocate full vocab sized maps, default 0 for special tokens
            tok2z = torch.zeros(space_vocab, dtype=torch.long)
            tok2y = torch.zeros(space_vocab, dtype=torch.long)
            tok2x = torch.zeros(space_vocab, dtype=torch.long)

            # Shift by one to leave room for 0 for SOS
            tok2x[1:1 + num_cells] = x
            tok2y[1:1 + num_cells] = y
            tok2z[1:1 + num_cells] = z

            self.register_buffer('tok2x', tok2x)
            self.register_buffer('tok2y', tok2y)
            self.register_buffer('tok2z', tok2z)
        else:
            self.x_embedding = None
            self.y_embedding = None
            self.z_embedding = None
        
        self.__init_layers()

        if len(self.particle_list) > 1:
            self.__build_LoRA_modules()

        self.SOS_token = 0
        self.EOS_token = space_vocab - 2  
        self.pad_token = space_vocab - 1  
        self.EOS_energy_token = energy_vocab - 2  
        self.energy_pad_token = energy_vocab - 1 

    def __build_LoRA_modules(self,energy_vocab=False):
        self.embedding_adapter = {}
        self.particle_lora = {} 
        self.vocab_LoRA = {}
        for particle in self.particle_list:
            if particle == self.base_model_type:
                # Base particle (e.g., gamma): no LoRA
                self.particle_lora[particle] = [None] * len(self.attn_heads)
            else:
                print(f"Creating LoRA modules for particle type: {particle}. LoRA_r={self.LoRA_r}, LoRA_alpha={self.LoRA_alpha} ")
                lora_list = nn.ModuleList([
                    LoRA(self.embed_dim, lora_r=self.LoRA_r, alpha=self.LoRA_alpha, drop_rate=self.drop_rates[0], device=self.device)
                    for _ in range(len(self.attn_heads))])
                self.particle_lora[particle] = lora_list

                embedding_adapter_list = nn.ModuleList([
                    ConditionedAdapter(self.embed_dim) for _ in range(2)])  # 2 embedding modules: token and energy
                self.embedding_adapter[particle] = embedding_adapter_list

                self.vocab_LoRA[particle] = nn.ModuleList([Vocab_LoRA(vocab_size=self.space_vocab, embed_dim=self.embed_dim, lora_r= self.LoRA_r, alpha=self.LoRA_alpha, drop_rate=self.drop_rates[0], device=self.device),
                                                          Vocab_LoRA(vocab_size=self.energy_vocab, embed_dim=self.embed_dim, lora_r= self.LoRA_r, alpha=self.LoRA_alpha, drop_rate=self.drop_rates[0], device=self.device)])
        

        for particle, lora_list in self.particle_lora.items():
            if isinstance(lora_list, nn.ModuleList):
                self.add_module(f"particle_lora_{particle}", lora_list)

        for particle, embedding_adapter_list in self.embedding_adapter.items():
            if isinstance(embedding_adapter_list, nn.ModuleList):
                self.add_module(f"embedding_adapter_{particle}", embedding_adapter_list)
        
        for particle, vocab_lora_module in self.vocab_LoRA.items():
            if isinstance(vocab_lora_module, nn.ModuleList):
                self.add_module(f"vocab_lora_{particle}", vocab_lora_module)


    def __init_layers(self,):
        # Can refactor this - fine for now
        layers_ = [CATransformerBlock(self.embed_dim,
                                      self.attn_heads[0],
                                      self.mlp_scale,
                                      drop_rate=self.drop_rates[0],
                                      use_MoE=self.use_MoE,
                                      num_experts=self.num_experts,
                                      num_classes=self.num_classes,
                                      device=self.device)]
        layers_ += [TransformerBlock(self.embed_dim,
                                     self.attn_heads[i],
                                     self.mlp_scale,
                                     drop_rate=self.drop_rates[i],
                                     use_MoE=self.use_MoE,
                                     num_experts=self.num_experts,
                                     num_classes=self.num_classes,
                                     device=self.device) for i in range(1, len(self.attn_heads))]
        self.layers = nn.ModuleList(layers_)

    def _allocate_kv_cache(self, batch_size, max_len, device, dtype=torch.float16):
        """
        Pre-allocate KV cache buffers to avoid concatenation overhead.
        
        Returns list of dicts, one per layer:
        [{'k': tensor, 'v': tensor, 'seq_len': 0}, ...]
        """
        cache_list = []
        
        for layer in self.layers:
            if isinstance(layer, CATransformerBlock):
                num_heads = layer.attn.num_heads
                head_dim = layer.attn.head_dim
            else:  # TransformerBlock
                num_heads = layer.attn.num_heads
                head_dim = layer.attn.head_dim
            
            # Pre-allocate cache: [B, H, max_len, D]
            cache_k = torch.zeros(
                batch_size, num_heads, max_len, head_dim,
                dtype=dtype, device=device
            )
            cache_v = torch.zeros(
                batch_size, num_heads, max_len, head_dim,
                dtype=dtype, device=device
            )
            
            cache_list.append({
                'k': cache_k,
                'v': cache_v,
                'seq_len': 0  # Tracks how many tokens are cached
            })
        
        return cache_list

    def extend_model(self, new_material_list,closest_expert=None,particle_type="e-"):
        # Extend model to an additional material/particle combo by adding experts and updating the router
        # This is done for a specific particle type e.g., G4_W_gamma
        # Assumes experts per class is constant 
        # See fine_tune.py for usage during fine-tuning; assumes loading of N-1 expert model.
        # Eventually want to comment out print statements - annoying in multi GPU training
        self.lora_newly_created = False

        if particle_type not in self.particle_list and particle_type != self.base_model_type:
            print("Adding new particle type; creating LoRA modules.")
            self.particle_list.append(particle_type)
            self.__build_LoRA_modules()
            self.lora_newly_created = True

        elif particle_type in self.particle_list and particle_type != self.base_model_type:
            print("Existing particle type; reusing LoRA modules.")
            # LoRA modules already loaded into model
            self.lora_newly_created = False
        elif particle_type == self.base_model_type:
            print("Base particle type; no LoRA modules added.")
            self.lora_newly_created = False
        else:
            raise ValueError("Unexpected particle type condition.")
        

        if not self.use_MoE:
            raise ValueError("Model is not using MoE; cannot extend materials.")

        if closest_expert is None:
            print("No closest expert specified; using last expert for new material/particle initialization.")

        current_num_classes = len(self.material_list)
        new_num_classes = len(new_material_list)

        if new_num_classes != current_num_classes + 1:
            print("Can only add one new material/particle at a time currently.")
            exit(1)
        
        if new_num_classes <= current_num_classes:
            print("No new materials to add. Something is likely wrong.")
            exit(1)

        # experts per class should be same for all classes
        experts_per_class = self.num_experts // current_num_classes
        new_num_experts = new_num_classes * experts_per_class

        old_material_list = self.material_list.copy()

        print(f"Extending from {self.material_list} to {new_material_list}")
        print(f"Current: {current_num_classes} classes * {experts_per_class} experts/class = {self.num_experts} total")
        print(f"New:     {new_num_classes} classes * {experts_per_class} experts/class = {new_num_experts} total")

        self.material_list = new_material_list
        self.num_classes = new_num_classes
        self.num_experts = new_num_experts

        for layer_idx, layer in enumerate(self.layers):
            if hasattr(layer, 'FF') and isinstance(layer.FF, MoE):
                old_moe = layer.FF
                current_num_experts = old_moe.num_experts
                
                print(f"Layer {layer_idx}: Extending MoE from {current_num_experts} -> {new_num_experts} experts")
                new_MoE = MoE(old_moe.embed_dim,
                              mlp_scale=old_moe.mlp_scale, 
                              num_experts=new_num_experts,
                              num_classes=new_num_classes,
                              drop_rate=old_moe.drop_rate)

                # Copy over existing experts
                for i in range(current_num_experts):
                    new_MoE.experts[i] = copy.deepcopy(old_moe.experts[i])
                    
                # Initialize new experts
                for j in range(current_num_experts, new_num_experts):
                    if closest_expert is not None:
                        closest_expert_idx = old_material_list.index(closest_expert) * experts_per_class + (j % experts_per_class)
                        print(f"Initializing new expert {j} from closest expert {closest_expert}, at index {closest_expert_idx}.")
                        new_MoE.experts[j] = copy.deepcopy(old_moe.experts[closest_expert_idx])
                    else:
                        print(f"Initializing new expert {j} randomly.")
                        # Random init
                        new_MoE.experts[j] = copy.deepcopy(old_moe.experts[-1])
                        for param in new_MoE.experts[j].parameters():
                            if param.requires_grad:
                                nn.init.normal_(param.data, mean=0.0, std=0.01)

                layer.FF = new_MoE

        print(f"Extended to {self.num_experts} experts for materials: {self.material_list}")

    def embed_space_tokens(self, idx, pos_idx):
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(pos_idx)

        if self.x_embedding is None:
            # No 3D embeddings
            return tok_emb + pos_emb
        
        # Map tokens -> z,y,x indices via precomputed lookup
        x_coord = self.tok2x[idx]
        y_coord = self.tok2y[idx]
        z_coord = self.tok2z[idx]

        x_emb = self.x_embedding(x_coord)
        y_emb = self.y_embedding(y_coord)
        z_emb = self.z_embedding(z_coord)

        return tok_emb + pos_emb + x_emb + y_emb + z_emb

    def forward(self, x, e, initial_energy, material_index,padding_mask=None,particle_type="gamma"):
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        if not self.digitize_energy:
            e_embed = self.energy_embedding(e.view(-1, 1)).view(batch_size, seq_len, -1)
        else:
            e_embed = self.energy_embedding(e)

        e_embed = e_embed + self.energy_pos_embedding(pos)

        e_embed = self.embedding_adapter[particle_type][1](e_embed) if (hasattr(self, 'embedding_adapter') and particle_type in self.embedding_adapter) else e_embed

        # Ensure initial_energy has shape (B, 1) before embedding
        if initial_energy.dim() == 1:
            initial_energy = initial_energy.unsqueeze(1)        # (B, 1)
        elif initial_energy.dim() == 2 and initial_energy.size(1) == 1:
            pass  # already (B, 1)
        else:
            initial_energy = initial_energy.view(batch_size, 1)  # force (B, 1)

        # Embed to (B, embed_dim) then add a time-step dimension -> (B, 1, embed_dim)
        initial_energy_embed = self.initial_energy_embedding(initial_energy).view(batch_size, 1, -1)  # (B, 1, E)

        # Apply adaptations to embeddings if applicable
        x = self.token_embedding(x) + self.pos_embedding(pos)
        x = self.embedding_adapter[particle_type][0](x) if (hasattr(self, 'embedding_adapter') and particle_type in self.embedding_adapter) else x

        e_embed = torch.cat((initial_energy_embed, e_embed), dim=1)  # Make sure to concat initial energy here
        x = torch.cat((initial_energy_embed, x), dim=1)

        # Instead of adding time and position embeddings, combine through Cross attention
        # Query from energy, given space (key, value)
        if padding_mask is not None:
            energy_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)  # No masking for kinematic tokens
            padding_mask = torch.cat((energy_mask, padding_mask), dim=1)


        if self.training:
            load_balance = 0.0
            for i, layer in enumerate(self.layers):
                lora_mod = None
                if hasattr(self, 'particle_lora') and particle_type in self.particle_lora:
                    lora_list = self.particle_lora[particle_type]
                    if isinstance(lora_list, (list, nn.ModuleList)) and len(lora_list) > 0:
                        lora_mod = lora_list[i] if lora_list[i] is not None else None
                
                if layer.__class__.__name__ == "CATransformerBlock":
                    x, _kv, load = layer(x, e_embed, material_index, padding_mask=padding_mask, classification=self.classification,LoRA_module=lora_mod)
                else:
                    x, _kv, load = layer(x, material_index, padding_mask=padding_mask, classification=self.classification,LoRA_module=lora_mod)
                load_balance += load
        else:
            for i, layer in enumerate(self.layers):
                lora_mod = None
                if hasattr(self, 'particle_lora') and particle_type in self.particle_lora:
                    lora_list = self.particle_lora[particle_type]
                    if isinstance(lora_list, (list, nn.ModuleList)) and len(lora_list) > 0:
                        lora_mod = lora_list[i] if lora_list[i] is not None else None
            
                if layer.__class__.__name__ == "CATransformerBlock":
                    x, _kv, _lb = layer(x, e_embed, material_index, padding_mask=padding_mask, classification=self.classification,LoRA_module=lora_mod)
                else:
                    x, _kv, _lb = layer(x, material_index, padding_mask=padding_mask, classification=self.classification,LoRA_module=lora_mod)

        x = self.LN(x)

        if not self.classification and not self.sequence_level:  # Generations - next hit prediction - sequence level will also be false then
            if not self.digitize_energy:
                e_out = self.energy_head(x).squeeze(-1)  # direct regression of time
            else:
                delta_e = self.vocab_LoRA[particle_type][1](x) if (hasattr(self, 'vocab_LoRA') and particle_type in self.vocab_LoRA) else 0.0
                e_out = self.energy_head(x) + delta_e  # logits over time

            delta_pixel = self.vocab_LoRA[particle_type][0](x) if (hasattr(self, 'vocab_LoRA') and particle_type in self.vocab_LoRA) else 0.0
            pixel = self.logits_head(x) + delta_pixel

            e_out = self.vocab_LoRA[particle_type][1].apply_product(e_out) if (hasattr(self, 'vocab_LoRA') and particle_type in self.vocab_LoRA) else e_out
            pixel = self.vocab_LoRA[particle_type][0].apply_product(pixel) if (hasattr(self, 'vocab_LoRA') and particle_type in self.vocab_LoRA) else pixel

            if self.training:
                return pixel, e_out, load_balance

            return pixel, e_out

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
    def generate(self, initial_energy, material_index, max_seq_len=2100,
                context_len=None, temperature: float = 1.0, method="Default",
                topK=100, nucleus_p=0.95, dynamic_temp=False, use_kv_cache=True,particle_type="gamma"):

        device = self.device
        B = initial_energy.shape[0]

        if initial_energy.dim() == 1:
            initial_energy = initial_energy.unsqueeze(1)
        init_e_embed = self.initial_energy_embedding(initial_energy).unsqueeze(1)

        is_done = torch.zeros(B, dtype=torch.bool, device=device)

        # Pre-allocate output tensors
        idx_buffer = torch.zeros((B, max_seq_len + 1), device=device, dtype=torch.long)
        if self.digitize_energy:
            e_buffer = torch.zeros((B, max_seq_len + 1), device=device, dtype=torch.long)
        else:
            e_buffer = torch.zeros((B, max_seq_len + 1), device=device, dtype=torch.float32)

        # Pre-allocate KV cache if enabled
        if use_kv_cache:
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                kv_caches = self._allocate_kv_cache(
                    batch_size=B,
                    max_len=max_seq_len + 2,  # +2 for init_energy + buffer
                    device=device,
                    dtype=AMP_DTYPE
                )
        else:
            kv_caches = [None] * len(self.layers)

        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            for step in range(max_seq_len):

                if use_kv_cache:
                    if step == 0:
                        # FIRST STEP: [B, 2, E]
                        idx_buffer[:, 0] = self.SOS_token
                        pos_idx = torch.zeros((B, 1), device=device, dtype=torch.long)
                        
                        sos_embed = self.token_embedding(idx_buffer[:, 0:1]) + self.pos_embedding(pos_idx)
                        # Adapter on embeddings if avail
                        sos_embed = self.embedding_adapter[particle_type][0](sos_embed) if (hasattr(self, 'embedding_adapter') and particle_type in self.embedding_adapter) else sos_embed
                        
                        x_t = torch.cat([init_e_embed, sos_embed], dim=1)
                        
                        if self.digitize_energy:
                            e_buffer[:, 0] = 0
                            e_sos = self.energy_embedding(e_buffer[:, 0:1]) + self.energy_pos_embedding(pos_idx)
                        else:
                            e_buffer[:, 0] = 0.0
                            e_sos = self.energy_embedding(e_buffer[:, 0:1].reshape(-1, 1)).view(B, 1, -1) + \
                                    self.energy_pos_embedding(pos_idx)
                        
                        # Adapter on embeddings if avail
                        e_sos = self.embedding_adapter[particle_type][1](e_sos) if (hasattr(self, 'embedding_adapter') and particle_type in self.embedding_adapter) else e_sos
                        e_t = torch.cat([init_e_embed, e_sos ], dim=1)
                        
                    else:
                        # SUBSEQUENT STEPS: [B, 1, E]
                        pos_idx = torch.full((B, 1), step, device=device, dtype=torch.long)
                        
                        x_t = self.token_embedding(idx_buffer[:, step:step+1]) + self.pos_embedding(pos_idx)
                        # Adapter on embeddings if avail
                        x_t = self.embedding_adapter[particle_type][0](x_t) if (hasattr(self, 'embedding_adapter') and particle_type in self.embedding_adapter) else x_t
                        
                        if self.digitize_energy:
                            e_t = self.energy_embedding(e_buffer[:, step:step+1]) + \
                                self.energy_pos_embedding(pos_idx)
                        else:
                            e_t = self.energy_embedding(e_buffer[:, step:step+1].reshape(-1, 1)).view(B, 1, -1) + \
                                self.energy_pos_embedding(pos_idx)

                        # Adapter on embeddings if avail
                        e_t = self.embedding_adapter[particle_type][1](e_t) if (hasattr(self, 'embedding_adapter') and particle_type in self.embedding_adapter) else e_t

                    h_t, kv_caches = self.forward_decode_step(
                        x_t, e_t, material_index,
                        padding_mask=None,
                        kv_caches=kv_caches,
                        is_first_step=(step == 0), particle_type=particle_type
                    )

                    delta_pixel = self.vocab_LoRA[particle_type][0](h_t) if (hasattr(self, 'vocab_LoRA') and particle_type in self.vocab_LoRA) else 0.0    
                    pixel_logits = (self.logits_head(h_t)[:, -1, :] + delta_pixel[:, -1, :]) 
                    pixel_logits = self.vocab_LoRA[particle_type][0].apply_product(pixel_logits) if (hasattr(self, 'vocab_LoRA') and particle_type in self.vocab_LoRA) else pixel_logits
                    pixel_logits = pixel_logits.squeeze(0) / temperature
                    if self.digitize_energy:
                        delta_e = self.vocab_LoRA[particle_type][1](h_t) if (hasattr(self, 'vocab_LoRA') and particle_type in self.vocab_LoRA) else 0.0
                        energy_logits = (self.energy_head(h_t)[:, -1, :] + delta_e[:, -1, :])
                        energy_logits = self.vocab_LoRA[particle_type][1].apply_product(energy_logits) if (hasattr(self, 'vocab_LoRA') and particle_type in self.vocab_LoRA) else energy_logits
                        energy_logits = energy_logits.squeeze(0) / temperature
                    else:
                        energy_val = self.energy_head(h_t)[:, -1]

                else:
                    # NO-CACHE path
                    if step == 0:
                        current_idx = torch.full((B, 1), self.SOS_token, device=device, dtype=torch.long)
                        if self.digitize_energy:
                            current_e = torch.zeros((B, 1), device=device, dtype=torch.long)
                        else:
                            current_e = torch.zeros((B, 1), device=device, dtype=torch.float32)
                    else:
                        current_idx = idx_buffer[:, :step+1].contiguous()
                        current_e = e_buffer[:, :step+1].contiguous()

                    pixel_all, e_out_all = self.forward(current_idx, current_e, initial_energy, material_index, padding_mask=None, particle_type=particle_type)

                    pixel_logits = pixel_all[:, -1, :] / temperature
                    if self.digitize_energy:
                        energy_logits = e_out_all[:, -1, :] / temperature
                    else:
                        energy_val = e_out_all[:, -1]

                # Sample tokens
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

                if self.digitize_energy:
                    probs_t = F.softmax(energy_logits, dim=-1, dtype=torch.float32)
                    if method == "Default":
                        e_next = torch.multinomial(probs_t, num_samples=1)
                    elif method == "TopK":
                        e_next = self.__topK(energy_logits, topK)
                    elif method == "Nucleus":
                        e_next = self.__nucleus(energy_logits, nucleus_p)
                    elif method == "Greedy":
                        e_next = torch.argmax(energy_logits, dim=-1, keepdim=True)
                    else:
                        e_next = self.__min_p(energy_logits)
                else:
                    e_next = energy_val.unsqueeze(1)

                # EOS handling
                if self.digitize_energy:
                    newly_done = (e_next.squeeze(1) == self.EOS_energy_token) | (idx_next.squeeze(1) == self.EOS_token)
                else:
                    newly_done = (idx_next.squeeze(1) == self.EOS_token)

                pad_mask = is_done & ~newly_done
                eos_mask = newly_done
                is_done |= newly_done

                # Store tokens
                idx_store = idx_next.squeeze(1).clone()
                idx_store[eos_mask] = self.EOS_token
                idx_store[pad_mask] = self.pad_token
                idx_buffer[:, step + 1] = idx_store

                if self.digitize_energy:
                    e_store = e_next.squeeze(1).clone()
                    e_store[eos_mask] = self.EOS_energy_token
                    e_store[pad_mask] = self.energy_pad_token
                    e_buffer[:, step + 1] = e_store
                else:
                    e_store = e_next.squeeze(1).clone()
                    if step > 0:
                        e_store[pad_mask] = e_buffer[:, step][pad_mask]
                    e_buffer[:, step + 1] = e_store

                if torch.all(is_done):
                    actual_len = step + 2
                    return idx_buffer[:, :actual_len].contiguous(), e_buffer[:, :actual_len].contiguous()

        return idx_buffer, e_buffer

    def forward_decode_step(self, x_t, e_t, material_index,
                            padding_mask=None, kv_caches=None, is_first_step=False, particle_type="gamma"):
        """
        Args:
            x_t: [B, T_new, E] where T_new=2 on first step, 1 after
            e_t: [B, T_new, E] where T_new=2 on first step, 1 after
            kv_caches: List of cache dicts, one per layer
            is_first_step: True only on step 0
        
        Returns:
            h_t: [B, 1, E] - output for newest position only
            updated_caches: List of updated cache dicts
        """
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)

        x = x_t
        e = e_t
        new_caches = []
        
        for i, (layer, cache) in enumerate(zip(self.layers, kv_caches)):
            lora_mod = None
            if hasattr(self, 'particle_lora') and particle_type in self.particle_lora:
                lora_list = self.particle_lora[particle_type]
                if isinstance(lora_list, (list, nn.ModuleList)) and len(lora_list) > 0:
                    lora_mod = lora_list[i] if lora_list[i] is not None else None

            if isinstance(layer, CATransformerBlock):
                x, updated_cache, _lb = layer(
                    x, e, material_index,
                    padding_mask=padding_mask,
                    classification=self.classification,
                    past_kv=cache,LoRA_module=lora_mod
                )
                new_caches.append(updated_cache)
            else:
                x, updated_cache, _lb = layer(
                    x, material_index,
                    padding_mask=padding_mask,
                    classification=self.classification,
                    past_kv=cache,LoRA_module=lora_mod
                )
                new_caches.append(updated_cache)

        x = self.LN(x)
        
        # Ensure output is [B, 1, E]
        x = x[:, -1:, :]
        
        return x, new_caches