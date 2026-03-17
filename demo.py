import os
import math
# Set these BEFORE importing torch
os.environ["TORCH_LOGS"] = "-cudagraphs" # Explicitly subtract cudagraphs from logs
os.environ["GLOG_minloglevel"] = "3"      # Suppress C++ Google Logs (3 = Fatal)

import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import h5py
from pathlib import Path
import pkbar
from datetime import datetime
import argparse

from plotting import (make_plots,visualize_vocab_LoRA,
                      plot_bias_comp,make_interactive_plots,
                      make_animated_event_viewer)

from models.GPT import ECAL_GPT
from dataloader.tokenizer import EnergyTokenizer
from dataloader.dataset import ECAL_Chunked_Dataset
from dataloader.dataloader import CreateInferenceLoader
from utils.utils import read_text, singular_value_checks

def main(config,args):

    if args.device == "cuda":
        device = 'cuda'
    else:
        device = 'cpu'

    vocab_size = config['model']['vocab_size']
    energy_vocab = config['model']['energy_vocab']
    embed_dim = config['model']['embed_dim']
    attn_heads = config['model']['attn_heads']
    num_blocks = config['model']['num_blocks']
    hidden_units = config['model']['hidden_units']
    mlp_scale = config['model']['mlp_scale']
    msl = config['model']['max_seq_length']
    drop_rates = config['model']['drop_rates']
    material_list = config['material_list']
    num_experts = len(material_list)
    use_MoE = bool(config['model']['use_MoE'])
    digitize_energy = bool(config['digitize_energy'])
    base_model_type = config['base_model_type']
    particle_list = config['particle_list']
    loRA_r = config['model']['LoRA_r']
    loRA_alpha = config['model']['LoRA_alpha']
    enable_head_LoRA = config['model']['enable_head_LoRA']
    enable_vocab_LoRA = config['model']['enable_vocab_LoRA']
    enable_embedding_adapter = config['model']['enable_embedding_adapter']
    vocab_LoRA_scale = config['model']['vocab_LoRA_scale']
    use_RoPE = config['model']['use_RoPE']
    learnable_vocabs = config['model']['learnable_vocabs']
    is_expanded = config['model']['is_expanded']

    energy_res = config['stats']['token_energy_res']
    e_max = config['stats']['token_energy_max']
    e_min = config['stats']['token_energy_min']
 
    energy_digitizer = EnergyTokenizer(e_max=e_max, e_min=e_min, resolution=energy_res)

    print("========= Generation Started =========")
    print("Device: ", args.device)
    print("Using AMP: ", args.use_amp)
    print("Temperature: ", args.temperature) if not args.dynamic_temp else print("Dynamic Temperature: Enabled")
    print("Generating showers for materials: ", material_list if args.material_to_generate is None else [args.material_to_generate])
    print("Cuda Graphs disabled: ", args.disable_cudagraphs)
    print("=====================================")

    if not args.disable_cudagraphs:
        print("Note: Only default method supported for CUDA graph generation.")

    print("Digitizing Energy - classification over adjacent vocabulary.")
    print("Energy vocab: ", config['model']['energy_vocab'])
    print("E_Max: ", e_max, " E_Min: ", e_min, "E_Res: ", energy_res)

    model = ECAL_GPT(vocab_size,
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
                material_list=material_list,
                base_model_type=base_model_type,
                particle_list=particle_list,
                LoRA_r=loRA_r,
                LoRA_alpha=loRA_alpha,
                enable_head_LoRA=enable_head_LoRA,
                enable_vocab_LoRA=enable_vocab_LoRA,
                enable_embedding_adapter=enable_embedding_adapter,
                vocab_LoRA_scale=vocab_LoRA_scale,
                learnable_vocabs=learnable_vocabs,
                use_RoPE=use_RoPE,is_expanded=is_expanded
                ).to(args.device)

    model_path = config['Inference']['model_path'] if args.model_path is None else args.model_path
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(checkpoint["net_state_dict"],strict=True)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Did you forget to add a material or particle to the config file?")
        exit(1)

    model.eval()

    try:
         model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    except Exception as _:
        # torch.compile may not be available / useful on this env; continue without it
        print('Could not compile model. Continuing...')
        pass
    global_e_max = config['stats']['global_energy_max']
    global_e_min = config['stats']['global_energy_min']
    stats = {"Initial_Energy_Max": global_e_max, "Initial_Energy_Min": global_e_min}

    sequence_lengths = {"gamma": 2100, "e-": 2700}

    event_dict = {}
    assert args.initial_energy >= global_e_min and args.initial_energy <= global_e_max, f"Initial energy {args.initial_energy} out of bounds [{global_e_min}, {global_e_max}]"
    initial_energy = ((args.initial_energy - global_e_min) / (global_e_max - global_e_min)) * 2 - 1.0
    initial_energy = torch.tensor([initial_energy]).unsqueeze(0).to(device).float()  # (1,1)

    if args.animated_viewer:
        assert args.material_to_generate in material_list, f"Material {args.material_to_generate} not found in material list. Check config and command line arguments."
        materials_to_generate = [args.material_to_generate]
        if not args.material_to_generate:
            print("Animated viewer enabled - define a single material to generate using --material_to_generate (e.g. --material_to_generate G4_W_gamma)")
    else:
        materials_to_generate = material_list

    for material in materials_to_generate:
        particle_type = "e-" if "e-" in material else "gamma"
        material_index = torch.tensor(material_list.index(material)).unsqueeze(0).to(device).long()
        gen_seq_len = sequence_lengths[particle_type]

        with torch.inference_mode():
            if args.use_amp:
                with autocast(dtype=torch.float16):     
                    if args.disable_cudagraphs:
                        generated_indices, generated_energies = model.generate(
                            initial_energy=initial_energy,
                            material_index=material_index,
                            method=args.sampling_method,
                            dynamic_temp=args.dynamic_temp,
                            max_seq_len=gen_seq_len,
                            temperature=args.temperature,
                            particle_type=particle_type    
                        )
                    else:
                        generated_indices, generated_energies = model.generate_with_cuda_graph(
                            initial_energy=initial_energy,
                            material_index=material_index,
                            max_seq_len=gen_seq_len,
                            temperature=args.temperature,
                            particle_type=particle_type,
                            dynamic_temp=args.dynamic_temp)
            else:
                if args.disable_cudagraphs:
                    generated_indices, generated_energies = model.generate(
                        initial_energy=initial_energy,
                        material_index=material_index,
                        method=args.sampling_method,
                        dynamic_temp=args.dynamic_temp,
                        max_seq_len=gen_seq_len,
                        temperature=args.temperature,
                        particle_type=particle_type    
                    )
                else:
                    generated_indices, generated_energies = model.generate_with_cuda_graph(
                        initial_energy=initial_energy,
                        material_index=material_index,
                        max_seq_len=gen_seq_len,
                        temperature=args.temperature,
                        particle_type=particle_type,
                        dynamic_temp=args.dynamic_temp)

            generated_indices = generated_indices.detach().cpu().numpy()
            generated_energies = generated_energies.detach().cpu().numpy()

        z,x,y,E = energy_digitizer.decode(generated_indices, generated_energies, material=material)
        energy_sum = E.sum()

        event_dict[material] = {"z": z, "x": x, "y": y, "E": E, "init_E": args.initial_energy, "energy_sum": energy_sum}

    if args.animated_viewer:
        viewer_path = make_animated_event_viewer(event_dict, output_dir="Animations",sort_by="energy")
        print(f"View the event at: {viewer_path}")
        viewer_path = make_animated_event_viewer(event_dict, output_dir="Animations",sort_by="z")
        print(f"View the event at: {viewer_path}")       

    else:
        make_interactive_plots(event_dict)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate showers using trained GPT model.")
    parser.add_argument('--config', type=str, default="config/config.json", help='Path to config JSON file.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: "cuda" or "cpu".')
    parser.add_argument('--initial_energy', type=float, default=50.0, help='Initial energy for generation.')
    parser.add_argument('--use_amp', action='store_true', help='Whether to use automatic mixed precision.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature during generation.')
    parser.add_argument('--dynamic_temp', action='store_true', help='Whether to use dynamic temperature during generation.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint for generation. Overrides config if set.')
    parser.add_argument('--disable_cudagraphs', action='store_true', help='Whether to disable CUDA graphs during generation. May help with stability on some environments.')
    parser.add_argument('--animated_viewer', action='store_true', help='Whether to create an animated viewer for the generated event.')
    parser.add_argument('--material_to_generate',default="G4_W_gamma",help='Specify a single material to generate (e.g. G4_W_gamma) for animated viewer.')
    args = parser.parse_args()

    os.makedirs("Animations", exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config, args)