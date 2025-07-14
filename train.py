import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

from utils.utils import energy_loss_fn

import os
import json
import argparse
import random
import numpy as np
import pkbar
import math
import warnings
from datetime import datetime

from dataloader.dataset import ECAL_Dataset
from dataloader.tokenizer import EnergyTokenizer
from dataloader.dataloader import CreateECALLoaders, CreateLoadersMoE

from models.GPT import ECAL_GPT

warnings.filterwarnings("ignore", message=".*weights_only.*")


def main(config, resume, distributed):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Running on CPU.")
    else:
        print(f"Running on {torch.cuda.device_count()} GPU(s).")

    if device.type == 'cuda':
        torch.cuda.manual_seed(config['seed'])


    # Create experiment name
    curr_date = datetime.now()
    exp_name = config['name'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    exp_name = exp_name[:-11]
    print(exp_name)

    # Create directory structure
    output_folder = config['output']['dir']
    os.makedirs(os.path.join(output_folder, exp_name), exist_ok=True)
    with open(os.path.join(output_folder, exp_name, 'config.json'), 'w') as outfile:
        json.dump(config, outfile)

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
    num_experts = config['model']['num_experts']
    num_classes = config['model']['num_classes']
    use_MoE = bool(config['model']['use_MoE'])

    # Time tokenization
    digitize_energy = bool(config['digitize_energy'])
    if digitize_energy:
        print("Digitizing Energy - classification over adjacent vocabulary.")
        print("Energy vocab: ", config['model']['energy_vocab'])
        energy_res = config['stats']['energy_res']
        e_max = config['stats']['energy_max']
        e_min = config['stats']['energy_min'] 
        print("E_Max: ", e_max, " E_Min: ", e_min, "E_Res: ", energy_res)
        energy_digitizer = EnergyTokenizer(e_max=e_max, e_min=e_min, resolution=energy_res)
    else:
        print("Using regression over energy domain.")
        energy_digitizer = None

    print('Creating Loaders.')

    if use_MoE:
        print("Conditional generation with MoE - singular generative model training.")

        data_path = config['dataset']['training']['data_path']
        val_data_path = config['dataset']['validation']['data_path']
        in_memory = config['dataset']['in_memory']

        train_dataset = ECAL_Dataset(data_path=data_path, max_seq_length=msl, energy_digitizer=energy_digitizer, in_memory=in_memory)
        val_dataset = ECAL_Dataset(data_path=val_data_path, max_seq_length=msl, energy_digitizer=energy_digitizer, in_memory=in_memory)
        train_loader, val_loader = CreateLoadersMoE(train_dataset, val_dataset, config)
    else:
        data_path = config['dataset']['training']['data_path']
        val_data_path = config['dataset']['validation']['data_path']
        in_memory = config['dataset']['in_memory']

        train_dataset = ECAL_Dataset(data_path=data_path, max_seq_length=msl, energy_digitizer=energy_digitizer, in_memory=in_memory)
        val_dataset = ECAL_Dataset(data_path=val_data_path, max_seq_length=msl, energy_digitizer=energy_digitizer, in_memory=in_memory)
        train_loader, val_loader = CreateECALLoaders(train_dataset, val_dataset, config)

    pad_token = train_dataset.pad_token
    EOS_token = train_dataset.EOS_token
    SOS_token = train_dataset.SOS_token

    energy_pad_token = train_dataset.energy_pad_token
    energy_EOS_token = train_dataset.energy_EOS_token

    print("========= Special Tokens ============")
    print(f"Cells - Pad: {pad_token}, SOS: {SOS_token}, EOS: {EOS_token}")
    print(f"Energy  - Pad: {energy_pad_token}, SOS: {SOS_token}, EOS: {energy_EOS_token}")
    print("=====================================")

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    run_val = True

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
                num_classes=num_classes,
                device=device)

    if device.type == 'cuda':
        if not distributed:
            print("Using single GPU.")
        else:
            print("Using {0} GPUs.".format(torch.cuda.device_count()))
            print(" ")
            net = DataParallel(net)

    t_params = sum(p.numel() for p in net.parameters())
    print("Network Parameters: ", t_params)
    net.to(device)

    # Optimizer
    if use_MoE:
        num_epochs = int(config['num_epochs_MoE'])
    else:
        num_epochs = int(config['num_epochs'])

    lr = float(config['optimizer']['lr'])

    # No need for warmup
    optimizer = torch.optim.RAdam(list(filter(lambda p: p.requires_grad, net.parameters())), lr=lr)

    startEpoch = 0
    global_step = 0

    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        startEpoch = dict['epoch'] + 1
        history = dict['history']
        global_step = dict['global_step']

        print('       ... Start at epoch:', startEpoch)
    else:
        print("========= Starting Training ================:")

    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      num_epochs:', num_epochs)
    print('')

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token)
    if digitize_energy:
        print("Energy vocab: ", energy_pad_token + 1)
        energy_ce = nn.CrossEntropyLoss(ignore_index=energy_pad_token)

    for epoch in range(startEpoch, num_epochs):

        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
        val_kbar = pkbar.Kbar(target=len(val_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
        ###################
        #  Training loop  #
        ###################
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            tokens = data[0].to(device).long()

            if use_MoE:
                class_label = data[-1].to(device).float()
            else:
                class_label = None

            next_tokens = tokens[:, 1:].clone()
            tokens = tokens[:, :-1]

            if not digitize_energy:
                energies = data[1].to(device).float()
            else:
                energies = data[1].to(device).long()

            next_energies = energies[:, 1:].clone()
            energies = energies[:, :-1]

            initial_energy = data[2].to(device).float()

            padding_mask = (tokens == pad_token).to(device, dtype=torch.bool)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                logits, e, load_balance = net(tokens, energies, initial_energy, class_label=class_label, padding_mask=padding_mask)

            # Slice off the prepended initial energy token
            logits = logits[:, 1:, :]
            e = e[:, 1:, :]

            pixel_loss = loss_fn(logits.reshape(-1, logits.size(-1)), next_tokens.reshape(-1))

            if not digitize_energy:
                regression_mask = ~torch.isin(next_tokens, torch.tensor([pad_token, SOS_token, EOS_token], device=next_tokens.device))
                energy_loss = energy_loss_fn(next_energies, e, regression_mask)
            else:
                energy_loss = energy_ce(e.reshape(-1, e.size(-1)), next_energies.reshape(-1))

            loss = pixel_loss + energy_loss + load_balance
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            # statistics
            running_loss += loss.item() * tokens.shape[0]

            kbar.update(i, values=[("loss", loss.item()),
                                   ("pix", pixel_loss.item()),
                                   ("energy", energy_loss.item()),
                                   ("load", load_balance.item())])

            global_step += 1

        history['train_loss'].append(running_loss / len(train_loader.dataset))

        ######################
        #  validation phase  #
        ######################
        if run_val:
            net.eval()
            val_energy_loss = 0.0
            val_pixel_loss = 0.0
            for i, data in enumerate(val_loader):
                tokens = data[0].to(device).long()

                if use_MoE:
                    class_label = data[-1].to(device).float()
                else:
                    class_label = None

                next_tokens = tokens[:, 1:].clone()
                tokens = tokens[:, :-1]

                if not digitize_energy:
                    energies = data[1].to(device).float()
                else:
                    energies = data[1].to(device).long()

                next_energies = energies[:, 1:].clone()
                energies = energies[:, :-1]

                initial_energy = data[2].to(device).float()

                padding_mask = (tokens == pad_token).to(device, dtype=torch.bool)

                with torch.no_grad():
                    logits, e = net(tokens, energies, initial_energy, class_label=class_label, padding_mask=padding_mask)

                # Slice off the prepended initial energy token
                logits = logits[:, 1:, :]
                e = e[:, 1:, :]

                if not digitize_energy:
                    regression_mask = ~torch.isin(next_tokens, torch.tensor([pad_token, SOS_token, EOS_token], device=next_tokens.device))
                    val_energy_loss += energy_loss_fn(next_energies, e, regression_mask)
                else:
                    val_energy_loss += energy_ce(e.reshape(-1, e.size(-1)), next_energies.reshape(-1))

                val_pixel_loss += loss_fn(logits.reshape(-1, logits.size(-1)), next_tokens.reshape(-1))

            val_energy_loss /= len(val_loader)
            val_pixel_loss /= len(val_loader)
            val_loss = val_pixel_loss + val_energy_loss

            kbar.add(1, values=[("Val_loss", val_loss.item()),
                                ("val_pix", val_pixel_loss.item()),
                                ("val_energy", val_energy_loss.item())])

            name_output_file = config['name'] + '_epoch{:02d}_val_loss_{:.6f}.pth'.format(epoch, val_loss)

        else:
            kbar.add(1, values=[('val_loss', 0.)])
            name_output_file = config['name'] + '_epoch{:02d}_train_loss_{:.6f}.pth'.format(epoch, running_loss / len(train_loader.dataset))

        filename = os.path.join(output_folder, exp_name, name_output_file)

        checkpoint = {}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint, filename)

        print('')


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Generative Training')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--distributed', default=0, type=int,
                        help='Training on multiple GPUs.')
    args = parser.parse_args()

    config = json.load(open(args.config))

    # os.makedirs("Trained_Models",exist_ok=True)

    main(config, args.resume, bool(args.distributed))