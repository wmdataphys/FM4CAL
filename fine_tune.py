import os
import sys 
import gc

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
import torch.multiprocessing as mp
import torch.distributed as dist

warnings.filterwarnings("ignore", message=".*weights_only.*")


def create_model(config,fine_tune_path=None,default_material_list=['G4_W_gamma','G4_Ta_gamma'],material_to_add=None,closest_expert=None):

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
    use_MoE = bool(config['model']['use_MoE'])
    digitize_energy = bool(config['digitize_energy'])

    if fine_tune_path is not None:
        print("Loading pre-trained model from: ", fine_tune_path)
        checkpoint = torch.load(fine_tune_path, map_location='cpu')
        state_dict = checkpoint['net_state_dict']
    else:
        state_dict = None
    
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
                material_list=materials_list)
                
    if state_dict:
        print("Loding state dict into model...")
        net.load_state_dict(state_dict,strict=False)


    net.extend_model(materials_list + [material_to_add], closest_expert=closest_expert)

    return net 

class Trainer:
    def __init__(self, config, rank, world_size, model,default_material_list=None,material_to_add=None):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.output_folder = config['output']['dir']
        self.exp_name = config['name']
        self.device = torch.device(f"cuda:{rank}")
        self.model = model.to(self.device)

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
        self.max_seq_length = config['model']['max_seq_length']
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
            print("=====================================")

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token)
        self.energy_ce = nn.CrossEntropyLoss(ignore_index=self.energy_pad_token) if self.digitize_energy else None
        self.energy_loss_fn = None  # you’ll need to define this or import it

        self.optimizer = torch.optim.RAdam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=float(config['optimizer']['lr']))
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
            
            # Freeze everything else (attention, embeddings, old experts, etc.)
            param.requires_grad = keep_trainable
            
            if keep_trainable:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
        
        if self.rank == 0:
            print(f"\n========= Parameter Freeze Status =========")
            print(f"Trainable params: {trainable_params:,}")
            print(f"Frozen params: {frozen_params:,}")
            print(f"Total params: {trainable_params + frozen_params:,}")
            print(f"Trainable %: {100 * trainable_params / (trainable_params + frozen_params):.2f}%")
            print(f"\nTrainable parameter names:")
            for name in trainable_names:
                print(f"  - {name}")
            print(f"==========================================\n")

    def init_kbar(self,num_files,num_epochs=1):
        total_samples = num_files * self.config['dataset']['tracks_per_file'] 
        total_batches = total_samples // self.config['dataloader']['train']['batch_size'] # // self.world_size
        self.kbar = pkbar.Kbar(target=total_batches, epoch=self.epoch, num_epochs=num_epochs, width=20)
            
    def load_chunked_dataset(self,file_list,verbose=False):
        global_e_max = self.stats['global_energy_max']
        global_e_min = self.stats['global_energy_min']
        stats = {"Initial_Energy_Max": global_e_max, "Initial_Energy_Min": global_e_min}
        dataset = ECAL_Chunked_Dataset(file_list=file_list,max_seq_length=self.max_seq_length,energy_digitizer=self.energy_digitizer
                                       ,verbose=verbose,ordering='energy',global_stats=stats,
                                       material_list=self.material_list)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        loader = CreateLoaderMoE(dataset, sampler=sampler, batch_size=self.config['dataloader']['train']['batch_size_cls'] // self.world_size,
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
                    logits, e, load_balance = self.model(tokens, energies, initial_energy, material_index=material_index, padding_mask=padding_mask)

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
                    logits, e, load_balance = self.model(tokens, energies, initial_energy, material_index, padding_mask=padding_mask)

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
        if val_loader:
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

                    logits,e = self.model(tokens, energies, initial_energy, material_index, padding_mask=padding_mask)
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

def run_worker(rank, world_size, config, all_train_files, all_val_files, fine_tune_path=None, default_material_list=None, material_to_add=None, closest_expert=None, run_val=True, write_path=None, checkpoint=None):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

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

    model = create_model(config, fine_tune_path=fine_tune_path, default_material_list=default_material_list, material_to_add=material_to_add, closest_expert=closest_expert)
    trainer = Trainer(config, rank, world_size, model,default_material_list=default_material_list,material_to_add=material_to_add)

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
        trainer.init_kbar(num_files,num_epochs)
        torch.cuda.empty_cache()  
        gc.collect()

        shuffled_files = np.array(all_train_files)[np.random.permutation(len(all_train_files))].tolist()
        
        for start_idx in range(0, num_files, chunk_size):
            file_chunk = shuffled_files[start_idx : start_idx + chunk_size]
            train_loader, sampler = trainer.load_chunked_dataset(file_chunk,verbose=False)
            
            if sampler is not None:
                sampler.set_epoch(epoch)

            #print("Starting training for epoch", epoch, "chunk", start_idx // chunk_size + 1)
            trainer.train_epoch(train_loader, sampler)

        if run_val:
            random_idx = np.random.randint(0, len(all_val_files), val_chunk_size)
            #print("Starting validation for epoch", epoch)
            val_loader, _ = trainer.load_chunked_dataset(all_val_files[random_idx].tolist(),verbose=False)
            trainer.on_epoch_end(val_loader,write_path)
        else:
            trainer.on_epoch_end(write_path)
    
    dist.destroy_process_group()


def read_text(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        lines = [line.strip() for line in lines]
        return lines

    except FileNotFoundError:
        raise ValueError(f"Error: The file '{file_path}' was not found.")

def main(config,default_material_list=["G4_W_gamma","G4_Ta_gamma"],fine_tune_path=None, material_to_add=None, closest_expert=None):
    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # Create experiment name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
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

    print("Default model trained on materials: ", default_material_list)
    print("Adding material for fine-tuning: ", material_to_add)


    train_files += read_text(config['dataset']['training'][material_to_add + '_train_files'])
    val_files += read_text(config['dataset']['validation'][material_to_add + '_val_files'])


    random.shuffle(train_files)
    random.shuffle(val_files)

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    run_worker(rank,world_size, config,train_files,val_files,
               write_path=write_path,
               fine_tune_path=fine_tune_path,
               default_material_list=default_material_list,
               material_to_add=material_to_add,
               closest_expert=closest_expert)

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Material based fine tuning.')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-m', '--default_material_list', nargs='+', default=["G4_W_gamma","G4_Ta_gamma"],
                        help='List of materials to include in training (default: ["G4_W_gamma","G4_Ta_gamma"])')
    parser.add_argument('--material_to_add', type=str, default="G4_Pb_gamma",
                        help='Material to add for fine-tuning (e.g., "G4_Pb_gamma")')                    
    parser.add_argument('--fine_tune_path', type=str, default=None,
                        help='Path to the pre-trained model checkpoint for fine-tuning')
    parser.add_argument('--closest_expert', type=str, default=None,
                        help='Closest expert material to initialize new expert from (e.g., "G4_Ta_gamma")')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.default_material_list,args.fine_tune_path,args.material_to_add,args.closest_expert)
