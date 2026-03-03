import os
import sys 
import gc
import time 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
# ✅ works across 1.10 → 2.x
from torch.cuda.amp import autocast, GradScaler

import json
import argparse
import random
import numpy as np
import pkbar    
import math
import warnings
from datetime import datetime

from dataloader.tokenizer import EnergyTokenizer
from dataloader.dataset import ECAL_Chunked_Dataset
from dataloader.dataloader import CreateLoaderMoE

from models.GPT import ECAL_GPT
from models.MoE import MoE
import torch.multiprocessing as mp
import torch.distributed as dist

warnings.filterwarnings("ignore", message=".*weights_only.*")


def create_model(config,fine_tune_path=None,default_material_list=['G4_W_gamma','G4_Ta_gamma'],material_to_add=None,
                 closest_expert=None,base_particle_list=None,particle_type=None,enable_pissa=False,new_seq_len=None):

    assert material_to_add is not None, "Material to add for fine-tuning must be specified."

    # Model params.
    vocab_size = config['model']['vocab_size']
    energy_vocab = config['model']['energy_vocab']
    embed_dim = config['model']['embed_dim']
    attn_heads = config['model']['attn_heads']
    num_blocks = config['model']['num_blocks']
    hidden_units = config['model']['hidden_units']
    mlp_scale = config['model']['mlp_scale']
    msl = config['model']['max_seq_length']
    drop_rates = config['model']['drop_rates']
    materials_list = default_material_list
    num_experts = len(materials_list)
    use_MoE = config['model']['use_MoE']
    digitize_energy = config['digitize_energy']
    loRA_r = config['model']['LoRA_r']
    loRA_alpha = config['model']['LoRA_alpha']
    enable_head_LoRA = config['model']['enable_head_LoRA']
    enable_vocab_LoRA = config['model']['enable_vocab_LoRA']
    learnable_vocabs = config['model']['learnable_vocabs']
    enable_embedding_adapter = config['model']['enable_embedding_adapter']
    vocab_LoRA_scale = config['model']['vocab_LoRA_scale']
    use_RoPE = config['model']['use_RoPE']
    is_expanded = config['model']['is_expanded']

    if fine_tune_path is not None:
        print("Loading pre-trained model from: ", fine_tune_path)
        checkpoint = torch.load(fine_tune_path, map_location='cpu')
        state_dict = checkpoint['net_state_dict']
    else:
        state_dict = None
    
    if base_particle_list is None:
        base_particle_list = config['particle_list']

    net = ECAL_GPT(vocab_size,
                   msl,
                   embed_dim,
                   attn_heads=attn_heads,
                num_blocks=num_blocks,
                hidden_units=hidden_units,
                digitize_energy=digitize_energy,
                mlp_scale=mlp_scale,
                energy_vocab=energy_vocab,
                drop_rates=drop_rates,
                use_MoE=use_MoE,
                num_experts=num_experts,
                material_list=materials_list,
                particle_list=base_particle_list,
                LoRA_r=loRA_r,
                LoRA_alpha=loRA_alpha,
                base_model_type=config['base_model_type'],
                enable_head_LoRA=enable_head_LoRA,
                enable_vocab_LoRA=enable_vocab_LoRA,
                learnable_vocabs=learnable_vocabs,
                enable_embedding_adapter=enable_embedding_adapter,
                vocab_LoRA_scale=vocab_LoRA_scale,
                use_RoPE=use_RoPE,
                is_expanded=is_expanded)
                # Base model - This is what we have trained on first, ever
                # If we fine tune from W_gamma -> W_e-, then base_model_type='gamma' but base_particle_list=['gamma','e-']
                # So model knows for e- to use LoRA + experts for e- and just experts for gamma if we fine tune to say G4_Pb_gamma
                


    if state_dict:
        print("Loding state dict into model...")
        net.load_state_dict(state_dict,strict=False)

    if enable_pissa:
        print("Using PiSSA weight initialization for vocab LoRA modules.")


    weights = {"logits_head": net.logits_head.weight.data,
                    "energy_head": net.energy_head.weight.data}


    experts_per_class = net.num_experts // len(default_material_list)
    net.extend_model(materials_list + [material_to_add], closest_expert=closest_expert, particle_type=particle_type, weights=weights, pissa_init=enable_pissa)

    if new_seq_len is not None:
        assert new_seq_len > msl, "New sequence length must be greater than the current maximum sequence length."
        
        print("Extending sequence length to:", new_seq_len)
        net.extend_sequence_length(new_seq_len)
        print("\n========= Sequence Adapater Check ==========")
        if not is_expanded:
            print("LPE Expansion doen't exist yet, comparing to existing positional embeddings.")
            old_e_weight = net.energy_pos_embedding.weight.data
            old_p_weight = net.pos_embedding.weight.data
        else:
            print("LPE expansion exists, comparing to expanded positional embeddings.")
            old_e_weight = net.lpe_expansion_energy.pos_embedding.weight.data
            old_p_weight = net.lpe_expansion_pos.pos_embedding.weight.data

        new_e_weight = net.lpe_expansion_energy.pos_embedding.weight[:old_e_weight.shape[0]].data
        new_p_weight = net.lpe_expansion_pos.pos_embedding.weight[:old_p_weight.shape[0]].data

        are_same_e = torch.allclose(old_e_weight, new_e_weight, rtol=1e-5)
        are_same_p = torch.allclose(old_p_weight, new_p_weight, rtol=1e-5)
        print(f"Energy Positional Embedding: New vs Old: {'MATCH' if are_same_e else 'DO NOT MATCH'}")
        print(f"Positional Embedding: New vs Old: {'MATCH' if are_same_p else 'DO NOT MATCH'}")
        print("=============================================")
    else:
        print("Using existing sequence length:", msl)



    print("\n========= New Expert Weight Check =========")
    for layer_idx, layer in enumerate(net.layers):
        if hasattr(layer, 'FF') and isinstance(layer.FF, MoE):
            if closest_expert is not None:
                source_idx = default_material_list.index(closest_expert) * experts_per_class
            else:
                source_idx = len(default_material_list) - 1  # Last old expert
            
            source_expert = layer.FF.experts[source_idx]
            new_expert = layer.FF.experts[-1]  # Last expert (newly added)
            
            # Compare first layer weights
            old_w = source_expert.nn[0].weight.data
            new_w = new_expert.nn[0].weight.data
            
            are_same = torch.allclose(old_w, new_w, rtol=1e-5)
            print(f"Layer {layer_idx}: New expert (idx={len(layer.FF.experts)-1}) vs Source expert '{closest_expert}' (idx={source_idx}): {'MATCH' if are_same else 'DO NOT MATCH'}")
    print("==========================================\n")

    if not net.enable_vocab_LoRA and net.learnable_vocabs:
        print("\n========= New Vocab Projection Check =========")
        
        if particle_type in net.vocab_LoRA and net.vocab_LoRA[particle_type][0] is not None:
            # Particle type already exists - compare to existing vocab heads
            print(f"Particle type '{particle_type}' already exists - comparing to existing vocab heads")
            old_space_w = net.vocab_LoRA[particle_type][0].weight.data
            old_space_b = net.vocab_LoRA[particle_type][0].bias.data
            old_ene_w = net.vocab_LoRA[particle_type][1].weight.data
            old_ene_b = net.vocab_LoRA[particle_type][1].bias.data
        else:
            # New particle type - compare to base model vocab heads
            print(f"New particle type '{particle_type}' - comparing to base model vocab heads")
            old_space_w = net.logits_head.weight.data
            old_space_b = net.logits_head.bias.data
            old_ene_w = net.energy_head.weight.data
            old_ene_b = net.energy_head.bias.data
        
        new_space_w = net.vocab_LoRA[particle_type][0].weight.data
        new_space_b = net.vocab_LoRA[particle_type][0].bias.data
        new_ene_w = net.vocab_LoRA[particle_type][1].weight.data
        new_ene_b = net.vocab_LoRA[particle_type][1].bias.data

        w_match = torch.allclose(old_space_w, new_space_w, rtol=1e-5)
        b_match = torch.allclose(old_space_b, new_space_b, rtol=1e-5)
        ew_match = torch.allclose(old_ene_w, new_ene_w, rtol=1e-5)
        eb_match = torch.allclose(old_ene_b, new_ene_b, rtol=1e-5)

        print(f"Space vocab Projection Weights: New vs Old: {'MATCH' if w_match else 'DO NOT MATCH'}")
        print(f"Space vocab Projection Bias: New vs Old: {'MATCH' if b_match else 'DO NOT MATCH'}")
        print(f"Energy vocab Projection Weights: New vs Old: {'MATCH' if ew_match else 'DO NOT MATCH'}")
        print(f"Energy vocab Projection Bias: New vs Old: {'MATCH' if eb_match else 'DO NOT MATCH'}")
        print("==========================================\n")

    return net 

class Trainer:
    def __init__(self, config, rank, world_size, model, default_material_list=None, material_to_add=None, particle_type="gamma",new_seq_len=None):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.output_folder = config['output']['dir']
        self.exp_name = config['name']
        self.device = torch.device(f"cuda:{rank}")
        self.model = model.to(self.device)
        self.particle_type = particle_type
        self.freeze_model(self.model)

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])

        self.use_amp = config['use_amp']
        if self.use_amp:
            self.scaler = GradScaler()

        self.use_MoE = bool(config['model']['use_MoE'])
        self.digitize_energy = bool(config['digitize_energy'])
        self.pad_token = config['special_tokens']['pad_token']
        self.energy_pad_token = config['special_tokens']['energy_pad_token']
        self.energy_EOS_token = config['special_tokens']['energy_EOS_token']
        self.SOS_token = config['special_tokens']['SOS_token']
        self.EOS_token = config['special_tokens']['EOS_token']
        self.stats = config['stats']
        self.max_seq_length = config['model']['max_seq_length'] if new_seq_len is None else new_seq_len
        # Pass through args, or config material list
        if default_material_list is None:
            self.material_list = config['material_list']
        else:   
            self.material_list = default_material_list

        self.material_to_add = material_to_add

        if self.material_to_add is not None:
            self.material_list.append(self.material_to_add)


        if self.rank == 0:
            print("========= Special Tokens ============")
            print(f"Pixels - Pad: {self.pad_token}, SOS: {self.SOS_token}, EOS: {self.EOS_token}")
            print(f"Energy   - Pad: {self.energy_pad_token}, SOS: {self.SOS_token}, EOS: {self.energy_EOS_token}")
            print(f"Max Sequence Length: {self.max_seq_length}")
            print("=====================================")

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token)
        self.energy_ce = nn.CrossEntropyLoss(ignore_index=self.energy_pad_token) if self.digitize_energy else None
        self.energy_loss_fn = None  # you’ll need to define this or import it

        vocab_lora_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if 'vocab_lora' in name and param.requires_grad:
                vocab_lora_params.append(param)
            elif param.requires_grad:
                other_params.append(param)
            else:
                pass  # Frozen parameters

        self.optimizer = torch.optim.AdamW([
            {'params': vocab_lora_params, 'weight_decay': config['optimizer']['decay_vocab']},  
            {'params': other_params, 'weight_decay': config['optimizer']['decay_else']}         
        ], lr=float(config['optimizer']['lr_ft']))

        if self.rank == 0:
            print("\n========= Optimizer Parameter Groups =========")
            print(f"Vocab LoRA params: {sum(p.numel() for p in vocab_lora_params):,} parameters with weight decay {config['optimizer']['decay_vocab']}")
            print(f"Other trainable params: {sum(p.numel() for p in other_params):,} parameters with weight decay {config['optimizer']['decay_else']}")
            print("=============================================\n")

        self.num_epochs = config['num_epochs']
        milestones = [5,10] 
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,gamma=0.1)
        print("Using LR Scheduler with milestones at epochs: ", milestones)
        self.history = {'train_loss': [], 'val_loss': []}
        self.global_step = 0
        self.epoch = 0
        self.global_batch_in_epoch = 0

        # Energy tokenization
        if self.digitize_energy:
            token_energy_res = config['stats']['token_energy_res']
            token_e_max = config['stats']['token_energy_max']
            token_e_min = config['stats']['token_energy_min'] 
            self.energy_digitizer = EnergyTokenizer(e_max=token_e_max, e_min=token_e_min, resolution=token_energy_res)

            if self.rank == 0:
                print("Digitizing Energy - classification over adjacent vocabulary.")
                print("Energy vocab: ", config['model']['energy_vocab'])
                print("Token_E_Max: ", token_e_max, " E_Min: ", token_e_min, "E_Res: ", token_energy_res)
        else:
            self.energy_digitizer = None
            if self.rank == 0:
                print("Using regression over energy domain.")

    def freeze_model(self, model):
        # Decide if we freeze LoRA as well
        use_lora = (self.particle_type != model.base_model_type)
        lora_exists = not model.lora_newly_created

        # Get the material list - last material is the new one
        new_material_idx = len(model.material_list) - 1
        num_classes = model.num_classes
        num_experts = model.num_experts
        experts_per_class = num_experts // num_classes
        
        # New expert indices for the last class
        new_expert_start_idx = new_material_idx * experts_per_class
        new_expert_end_idx = (new_material_idx + 1) * experts_per_class
        
        if self.rank == 0:
            print(f"\n========= Fine-tuning Configuration =========")
            print(f"Total materials: {model.material_list}")
            print(f"New material: {model.material_list[-1]} (index {new_material_idx})")
            print(f"Experts per class: {experts_per_class}")
            print(f"New expert indices: [{new_expert_start_idx}, {new_expert_end_idx})")
            print(f"Base particle: {model.base_model_type} | Fine-tune particle: {self.particle_type}")
            print(f"Use LoRA: {use_lora} | LoRA exists (will freeze): {lora_exists and use_lora}")
            print(f"============================================\n")

        frozen_params = 0
        trainable_params = 0
        trainable_names = []
        
        for name, param in model.named_parameters():
            keep_trainable = False
            
            # Only train new experts (e.g., "experts.2" for 3rd expert)
            for expert_idx in range(new_expert_start_idx, new_expert_end_idx):
                if f"experts.{expert_idx}." in name:
                    keep_trainable = True
                    trainable_names.append(name)
                    break

            # Allow router to have gradients, but we register a masking hook
            # only updated along new expert directions
            if "router." in name:
                keep_trainable = True
                trainable_names.append(name)
            
            # Check LoRA freeze condition
            if "particle_lora" in name:
                if lora_exists:
                    # LoRA already trained, freeze it for new materials
                    keep_trainable = False
                else:
                    # First time training LoRA for this particle
                    keep_trainable = True
                    trainable_names.append(name)

            # Check init_e_adapter freeze condition
            if "init_e_adapter" in name:
                if lora_exists:
                    # Adapter already trained, freeze it for new materials
                    keep_trainable = False
                else:
                    # First time training adapter for this particle
                    keep_trainable = True
                    trainable_names.append(name)

            # Check Embedding LoRA freeze condition
            if "embedding_adapter" in name:
                if lora_exists:
                    # LoRA already trained, freeze it for new materials
                    keep_trainable = False
                else:
                    # First time training LoRA for this particle
                    keep_trainable = True
                    trainable_names.append(name)
            
            # Check Vocab LoRA freeze condition
            if "vocab_lora" in name:
                if lora_exists:
                    # LoRA already trained, freeze it for new materials
                    keep_trainable = False
                else:
                    # First time training LoRA for this particle
                    keep_trainable = True
                    trainable_names.append(name)

            if "lpe_expansion" in name:
                if hasattr(self.model, "lpe_newly_created") and self.model.lpe_newly_created:
                    # LPE newly created, train it for new materials
                    keep_trainable = True
                    trainable_names.append(name)
                else:
                    # LPE already trained, freeze it for new materials
                    keep_trainable = False

            # Freeze everything else (attention, embeddings, old experts, etc.)
            param.requires_grad = keep_trainable
            
            if keep_trainable:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
                
        
        if self.rank == 0:
            print(f"\n========= Parameter Freeze Status =========")
            print(f"Trainable params: {trainable_params:,}")
            print(f"Frozen params: {frozen_params:,}") # This is the base model
            print(f"Total params: {trainable_params + frozen_params:,}")
            print(f"Trainable %: {100 * trainable_params / (frozen_params):.2f}%")
            print(f"\nTrainable parameter names:")
            for name in trainable_names:
                print(f"  - {name}")
            print(f"==========================================\n")

    def init_kbar(self, num_files, num_epochs=1,n_events=None):
        total_samples = num_files * self.config['dataset']['tracks_per_file']

        if n_events is not None:
            total_samples = min(total_samples, n_events)

        per_gpu_bs = self.config['dataloader']['train']['batch_size_ft'] // self.world_size
        total_batches = math.ceil((total_samples / self.world_size) / per_gpu_bs)
        self.kbar = pkbar.Kbar(target=total_batches, epoch=self.epoch, num_epochs=num_epochs, width=20)
            
    def load_chunked_dataset(self,file_list,verbose=False,n_events=None,dataset_seed=42): 
        global_e_max = self.stats['global_energy_max']
        global_e_min = self.stats['global_energy_min']
        stats = {"Initial_Energy_Max": global_e_max, "Initial_Energy_Min": global_e_min}
        dataset = ECAL_Chunked_Dataset(file_list=file_list,max_seq_length=self.max_seq_length,energy_digitizer=self.energy_digitizer
                                       ,verbose=verbose,ordering='energy',global_stats=stats,
                                       material_list=self.material_list,n_events=n_events,dataset_seed=dataset_seed)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        loader = CreateLoaderMoE(dataset, sampler=sampler, batch_size=self.config['dataloader']['train']['batch_size_ft'] // self.world_size,
                                num_workers=self.config['dataloader']['train']['num_workers'],
                                pin_memory=False,persistent_workers=False,prefetch_factor=self.config['dataloader']['train']['prefetch_factor'])
        return loader, sampler

    def train_epoch(self, train_loader, sampler):
        sampler.set_epoch(self.epoch)
        self.model.train()
        running_loss = 0.0


        for i, data in enumerate(train_loader):
            tokens = data[0].to(self.device).long()
            energies = data[1].to(self.device).long() if self.digitize_energy else data[1].to(self.device).float()
            initial_energy = data[2].to(self.device).float()
            material_index = data[-1].to(self.device).long() if self.use_MoE else None
            skip_idx = 1 if self.use_MoE else 1

            next_tokens = tokens[:, 1:].clone()
            tokens = tokens[:, :-1]

            next_energies = energies[:, 1:].clone()
            energies = energies[:, :-1]

            padding_mask = (tokens == self.pad_token).to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(dtype=torch.float16):
                    logits, e, load_balance = self.model(tokens, energies, initial_energy, material_index=material_index, padding_mask=padding_mask, particle_type=self.particle_type)

                    logits = logits[:, skip_idx:, :]
                    e = e[:, skip_idx:, :]

                    pixel_loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), next_tokens.reshape(-1))

                    if not self.digitize_energy:
                        regression_mask = ~torch.isin(
                            next_tokens,
                            torch.tensor([self.pad_token, self.SOS_token, self.EOS_token],
                                        device=next_tokens.device)
                        )
                        energy_loss = self.energy_loss_fn(next_energies, e, regression_mask)
                    else:
                        energy_loss = self.energy_ce(e.reshape(-1, e.size(-1)), next_energies.reshape(-1))

                    loss = pixel_loss + energy_loss + load_balance
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            else:
                with torch.set_grad_enabled(True):
                    logits, e, load_balance = self.model(tokens, energies, initial_energy, material_index, padding_mask=padding_mask, particle_type=self.particle_type)

                # Slice off the prepended initial energy token
                logits = logits[:, skip_idx:, :]
                e = e[:, skip_idx:, :]

                pixel_loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), next_tokens.reshape(-1))

                if not self.digitize_energy:
                    regression_mask = ~torch.isin(next_tokens, torch.tensor([self.pad_token, self.SOS_token, self.EOS_token], device=next_tokens.device))
                    energy_loss = self.energy_loss_fn(next_energies, e, regression_mask)
                else:
                    energy_loss = self.energy_ce(e.reshape(-1, e.size(-1)), next_energies.reshape(-1))

                loss = pixel_loss + energy_loss + load_balance
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            running_loss += loss.item() * tokens.size(0)

            with torch.no_grad():
                losses = torch.tensor([loss.item(), pixel_loss.item(), energy_loss.item(), load_balance.item()],
                                    device=self.device)

                dist.all_reduce(losses, op=dist.ReduceOp.SUM)
                losses /= self.world_size

            if self.rank == 0:
                # We only update for rank 0
                # So it looks like were missing half the data, its just averaged across all workers.
                # With 2 GPUs it will half the total batches, 4 GPUs its a quarter, etc.
                self.kbar.update(self.global_batch_in_epoch, values=[("loss", losses[0].item()), ("pix", losses[1].item()), ("energy", losses[2].item()), ("load", losses[3].item())])

            self.global_batch_in_epoch += 1
            self.global_step += 1

        epoch_loss = running_loss / len(train_loader.dataset)
        self.history['train_loss'].append(epoch_loss)

    def on_epoch_end(self, val_loader=None,write_path=None):
        if val_loader is not None:
            self.model.eval()
            val_pixel_loss = 0.0
            val_energy_loss = 0.0
            skip_idx = 1 if self.use_MoE else 1
            
            with torch.no_grad():
                for i,data in enumerate(val_loader):
                    tokens = data[0].to(self.device).long()
                    energies = data[1].to(self.device).long() if self.digitize_energy else data[1].to(self.device).float()
                    initial_energy = data[2].to(self.device).float()
                    material_index = data[-1].to(self.device).long() if self.use_MoE else None

                    next_tokens = tokens[:, 1:].clone()
                    tokens = tokens[:, :-1]

                    next_energies = energies[:, 1:].clone()
                    energies = energies[:, :-1]

                    padding_mask = (tokens == self.pad_token).to(self.device)

                    logits,e,_ = self.model(tokens, energies, initial_energy, material_index, padding_mask=padding_mask,particle_type=self.particle_type)
                    logits = logits[:, skip_idx:, :]
                    e = e[:, skip_idx:, :]

                    if self.digitize_energy:
                        val_energy_loss += self.energy_ce(e.reshape(-1, e.size(-1)), next_energies.reshape(-1))
                    else:
                        regression_mask = ~torch.isin(next_tokens, torch.tensor([self.pad_token, self.SOS_token, self.EOS_token], device=next_tokens.device))
                        val_energy_loss += self.energy_loss_fn(next_energies, e, regression_mask)

                    val_pixel_loss += self.loss_fn(logits.reshape(-1, logits.size(-1)), next_tokens.reshape(-1))

            val_energy_loss /= len(val_loader)
            val_pixel_loss /= len(val_loader)
            val_loss = val_pixel_loss + val_energy_loss
            self.history['val_loss'].append(val_loss.item())
        else:
            val_loss = torch.tensor(0.0)
            val_pixel_loss = torch.tensor(0.0)
            val_energy_loss = torch.tensor(0.0)

        with torch.no_grad():
            losses = torch.tensor([val_loss.item(), val_pixel_loss.item(), val_energy_loss.item()],
                                device=self.device)

            dist.all_reduce(losses, op=dist.ReduceOp.SUM)
            losses /= self.world_size

        if self.rank == 0:
            self.kbar.update(self.kbar.target, values=[("Val_loss", losses[0].item()),("val_pix",losses[1].item()),("val_ene",losses[2].item())])
            print()          
            sys.stdout.flush()  
            filename = os.path.join(write_path, f'Epoch{self.epoch:02d}_loss_{val_loss:.6f}.pth')
            torch.save({
                'net_state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'history': self.history,
                'global_step': self.global_step,
            }, filename)

def run_worker(rank, world_size, config, all_train_files, all_val_files, fine_tune_path=None, default_material_list=None, material_to_add=None, closest_expert=None, run_val=False, write_path=None, checkpoint=None,
               base_particle_list=["gamma"], particle_type='gamma', enable_pissa=False, n_events=None,dataset_seed=42,new_seq_len=None):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    print("Rank ", rank, " - dataset_seed: ", dataset_seed)
    # Array so we can Shuffle for permutations later - only val over diff subset each time
    all_val_files = np.array(all_val_files)

    num_epochs = config['num_epochs']
    chunk_size = config['dataloader']['train']['chunk_size']
    val_chunk_size = config['dataloader']['val']['chunk_size']

    if chunk_size > len(all_train_files):
        chunk_size = len(all_train_files)

    if val_chunk_size > len(all_val_files):
        val_chunk_size = len(all_val_files)

    num_files = len(all_train_files) 

    print(f"Rank {rank} - Starting training with {num_files} files, chunk size: {chunk_size}, num epochs: {num_epochs}")

    model = create_model(config, fine_tune_path=fine_tune_path, default_material_list=default_material_list, material_to_add=material_to_add, closest_expert=closest_expert,
                         base_particle_list=base_particle_list, particle_type=particle_type,enable_pissa=enable_pissa,new_seq_len=new_seq_len)
    trainer = Trainer(config, rank, world_size, model, default_material_list=default_material_list, material_to_add=material_to_add, 
                      particle_type=particle_type,new_seq_len=new_seq_len)

    if checkpoint is not None:
        if 'net_state_dict' in checkpoint:
            trainer.model.module.load_state_dict(checkpoint['net_state_dict'])
            print(f"Rank {rank} - Loaded model state from checkpoint.")
        if 'optimizer' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            trainer.epoch = checkpoint.get('epoch', 0)
            trainer.history = checkpoint.get('history', trainer.history)
            trainer.global_step = checkpoint.get('global_step', 0)
            print(f"Rank {rank} - Loaded optimizer state from checkpoint, starting at epoch {trainer.epoch}.")
        else:
            trainer.epoch = 0
            trainer.global_step = 0
            trainer.history = {'train_loss': [], 'val_loss': []}
    
    trainer.global_batch_in_epoch = 0

    for epoch in range(trainer.epoch,num_epochs):
        trainer.epoch = epoch
        trainer.global_batch_in_epoch = 0  
        trainer.init_kbar(num_files,num_epochs,n_events=n_events)
        torch.cuda.empty_cache()  
        gc.collect()

        if rank == 0:
            print("Learning rate: ", trainer.scheduler.get_last_lr()[0])

        shuffled_files = np.array(all_train_files)[np.random.permutation(len(all_train_files))].tolist()
        
        for start_idx in range(0, num_files, chunk_size):
            file_chunk = shuffled_files[start_idx : start_idx + chunk_size]
            train_loader, sampler = trainer.load_chunked_dataset(file_chunk,verbose=False, n_events=n_events, dataset_seed=dataset_seed)
            
            if sampler is not None:
                sampler.set_epoch(epoch)

            trainer.train_epoch(train_loader, sampler)

        trainer.scheduler.step()

        if run_val:
            random_idx = np.random.randint(0, len(all_val_files), val_chunk_size)
            #print("Starting validation for epoch", epoch)
            val_loader, _ = trainer.load_chunked_dataset(all_val_files[random_idx].tolist(),verbose=False)
            trainer.on_epoch_end(val_loader,write_path)
        else:
            trainer.on_epoch_end(val_loader=None, write_path=write_path)

    if rank == 0:
        final_ckpt = os.path.join(write_path, f'Epoch{trainer.epoch:02d}_loss_*.pth')
        import glob
        final_ckpt_files = glob.glob(final_ckpt)
        if final_ckpt_files:
            print(f"\n{'='*60}")
            print(f"FINAL_CHECKPOINT_PATH={final_ckpt_files[0]}")
            print(f"{'='*60}\n")
    
    dist.destroy_process_group()


def read_text(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        lines = [line.strip() for line in lines]
        return lines

    except FileNotFoundError:
        raise ValueError(f"Error: The file '{file_path}' was not found.")

def main(config,default_material_list=["G4_W_gamma","G4_Ta_gamma"],fine_tune_path=None, material_to_add=None, 
         closest_expert=None, base_particle_list=["gamma"], particle_type="gamma", enable_pissa=False, n_events=None,dataset_seed=42,new_seq_len=None):
    
    if n_events is not None:
        print(f"Fine-tuning on a subset of {n_events} events.")
        print(f"Dataset will use seed: {dataset_seed} (unique per run)")

    print("Setting random seed to: ", config['seed'])

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Create experiment name
    if n_events is not None:
        temp_ = f"___subset_{n_events}_events"
    else:
        temp_ = ""
        
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S') + temp_
    exp_name = exp_name[:-11]
    print(exp_name)

    # Create directory structure
    output_folder = config['output']['dir']
    os.makedirs(os.path.join(output_folder,exp_name),exist_ok=True)
    write_path = os.path.join(output_folder,exp_name)
    with open(os.path.join(output_folder,exp_name,'config.json'),'w') as outfile:
        json.dump(config, outfile)

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    train_files = []
    val_files = []

    print("Default model trained on material(s): ", default_material_list)
    print("Default model trained on particle type(s): ", base_particle_list)
    print("Adding material for fine-tuning: ", material_to_add)
    print("Particle type(s) for fine-tuning: ", particle_type)
    print("Closest expert for initialization: ", closest_expert)

    train_files += read_text(config['dataset']['training'][material_to_add + '_train_files'])
    val_files += read_text(config['dataset']['validation'][material_to_add + '_val_files'])

    if args.n_events is not None:
        # 10k is rough limit, this currently breaks at multiples of 10k
        n_events_per_file = config['dataset']['tracks_per_file']
        n_files_needed = math.ceil(args.n_events / n_events_per_file)
        
        if args.n_events % n_events_per_file == 0:
            n_files_needed += 1  # To ensure we have enough events

        random_idx = np.random.permutation(len(train_files))[:n_files_needed]
        train_files = [train_files[i] for i in random_idx]


    random.shuffle(train_files)
    random.shuffle(val_files)

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    run_worker(rank,world_size, config,train_files,val_files,
               write_path=write_path,
               fine_tune_path=fine_tune_path,
               default_material_list=default_material_list,
               material_to_add=material_to_add,
               closest_expert=closest_expert,
               base_particle_list=base_particle_list,
               particle_type=particle_type,
               enable_pissa=enable_pissa,
               n_events=n_events,
               dataset_seed=dataset_seed,
               new_seq_len=new_seq_len)

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Material based fine tuning.')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-m', '--default_material_list', nargs='+', default=["G4_W_gamma","G4_Ta_gamma"],
                        help='List of materials to include in pre-training (default: ["G4_W_gamma","G4_Ta_gamma"])')
    parser.add_argument('--material_to_add', type=str, default="G4_Pb_gamma",
                        help='Material to add for fine-tuning (e.g., "G4_Pb_gamma,G4_W_e-")')
    parser.add_argument('--base_particle_list', nargs='+', default=["gamma"],
                        help='List of particle types the base model was pre-trained on (default: ["gamma"])')
    parser.add_argument('--particle_type', type=str, default="gamma",help='Particle type to add for fine-tuning (e.g., "gamma,e-")')
    parser.add_argument('--fine_tune_path', type=str, default=None,
                        help='Path to the pre-trained model checkpoint for fine-tuning')
    parser.add_argument('--closest_expert', type=str, default=None,
                        help='Closest expert material to initialize new expert from (e.g., "G4_Ta_gamma")')
    parser.add_argument('--enable_pissa', action='store_true',
                        help='Enable PiSSA weight initialization for vocab LoRA modules.')
    parser.add_argument('--n_events', type=int, default=None, help='Number of events to use for fine-tuning.')
    parser.add_argument('--dataset_seed', type=int, default=42, help='Random seed for dataset shuffling (default: 42)')
    parser.add_argument('--new_seq_len', type=int, default=None, help='New sequence length for fine-tuning (default: None)')

    args = parser.parse_args()
    config = json.load(open(args.config))

    main(config,args.default_material_list,args.fine_tune_path,args.material_to_add,args.closest_expert,args.base_particle_list,args.particle_type,args.enable_pissa,args.n_events,args.dataset_seed,args.new_seq_len)
