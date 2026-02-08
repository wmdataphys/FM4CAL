import os

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

from plotting import make_plots,visualize_vocab_LoRA

from models.GPT import ECAL_GPT
from dataloader.tokenizer import EnergyTokenizer
from dataloader.dataset import ECAL_Chunked_Dataset
from dataloader.dataloader import CreateInferenceLoader
from utils.utils import read_text, singular_value_checks

def main(config,args):

    if args.num_showers != -1:
        print("\n" + "="*60)
        print("WARNING".center(60))
        print("="*60)
        print("Plotting functionality may not work as intended given")
        print("limited number of showers and materials.")
        print("Will refine in future iterations.")
        print("="*60 + "\n")

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
    use_kv_cache = bool(args.use_kv_cache)
    base_model_type = config['base_model_type']
    particle_list = config['particle_list']
    loRA_r = config['model']['LoRA_r']
    loRA_alpha = config['model']['LoRA_alpha']
    enable_head_LoRA = config['model']['enable_head_LoRA']
    enable_vocab_LoRA = config['model']['enable_vocab_LoRA']
    enable_embedding_adapter = config['model']['enable_embedding_adapter']
    vocab_LoRA_scale = config['model']['vocab_LoRA_scale']
    use_RoPE = config['model']['use_RoPE']

    energy_res = config['stats']['token_energy_res']
    e_max = config['stats']['token_energy_max']
    e_min = config['stats']['token_energy_min']

    if args.materials_to_generate is not None:
        materials_to_generate = args.materials_to_generate
        print("Generating for specified materials: ", materials_to_generate)
    else:
        materials_to_generate = material_list
        print("Generating for all materials in config: ", materials_to_generate)
    
    print(material_list)
    energy_digitizer = EnergyTokenizer(e_max=e_max, e_min=e_min, resolution=energy_res)
    outfile = os.path.join("Generations", config['Inference']['output_file'])

    if os.path.exists(outfile):
        print(f"Output file {outfile} already exists, do you want to overwrite it? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            print("Generation aborted by user, running plotting code only.")
            make_plots(outfile, energy_digitizer,materials_to_plot=materials_to_generate, num_showers=args.num_showers,material_list=material_list,comparison_path=args.comp_paths)
            exit(0)

    print("========= Generation Started =========")
    print("Device: ", args.device)
    print("Using AMP: ", args.use_amp)
    print("Using KV Cache: ", args.use_kv_cache)
    print("Sampling method: ", args.sampling_method)
    print("Temperature: ", args.temperature) if not args.dynamic_temp else print("Dynamic Temperature: Enabled")
    print("Generating showers for materials: ", materials_to_generate)
    print("Number of showers to generate: ", args.num_showers)
    print("=====================================")

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
                use_RoPE=use_RoPE
                ).to(args.device)

    model_path = config['Inference']['model_path']
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["net_state_dict"],strict=True)

    scaler = None
    if args.use_amp:
        scaler = GradScaler()
        if "scaler" in checkpoint:  # load scaler state if present
            scaler.load_state_dict(checkpoint["scaler"])
            print("Loaded AMP GradScaler state from checkpoint.")
        else:
            print("No GradScaler state found in checkpoint; starting fresh.")

    startEpoch = checkpoint.get("epoch", -1) + 1
    history = checkpoint.get("history", {})
    global_step = checkpoint.get("global_step", 0)
    print(f"Loaded from: {model_path} at epoch {startEpoch}, global_step {global_step}")

    model.eval()


    # for name, param in model.named_parameters():
    #     if "lora" in name:
    #         print(f"Parameter {name} has value: {param.data.cpu().numpy()}")

    # (Optional) compile AFTER loading. If you stay on CPU, compiling may not help; feel free to skip.
    try:
        if use_kv_cache == True:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    except Exception as _:
        # torch.compile may not be available / useful on this env; continue without it
        print('Could not compile model. Continuing...')
        pass

    test_files = []
    for material in materials_to_generate:
        # e.g., material = "G4_W_gamma" -> config['dataset']['test']['G4_W_gamma_test_files']
        # e.g., material = "G4_W_electron" -> config['dataset']['test']['G4_W_electron_test_files']
        test_files += read_text(config['dataset']['testing'][material + '_test_files'])
        if "e-" in material:
            particle_type = "e-"
        elif "gamma" in material:
            particle_type = "gamma"
        else:
            raise ValueError("Unknown particle type in material: ", material)

    for name, param in model.named_parameters():
        if particle_type in name:
            print(f"Parameter {name} has value: {param.mean().data.cpu().numpy()}")

    for name, module in model.named_modules():
        if particle_type in name:
            if "particle_lora" in name and hasattr(module, 'lora_B_Q'):
                # (A, B, r, alpha, layer_name, vocab=False):
                print(f"\n=== SVD Check: {name} ===")
                singular_value_checks(module.lora_A_Q, module.lora_B_Q, module.lora_r, module.scale, f"{name}.Q")
                singular_value_checks(module.lora_A_K, module.lora_B_K, module.lora_r, module.scale, f"{name}.K")
                singular_value_checks(module.lora_A_V, module.lora_B_V, module.lora_r, module.scale, f"{name}.V")
                singular_value_checks(module.lora_A_c_proj, module.lora_B_c_proj, module.lora_r, module.scale, f"{name}.c_proj")

            elif "vocab_lora" in name and hasattr(module, 'lora_B_vocab'):
                print(f"\n=== SVD Check: {name} ===")
                singular_value_checks(module.lora_A_vocab, module.lora_B_vocab, module.lora_r, module.scale, f"{name}", vocab=True)

                if args.visualize_vocab_LoRA:
                    if '0' in name:
                        label = "Pixel"
                        original_W = model.logits_head.weight.data
                    elif '1' in name:
                        label = "Energy"
                        original_W = model.energy_head.weight.data
                    else:
                        label = "Unknown"
                        print("Unknown vocab LoRA label for visualization. Exiting.")
                        exit(1)

                    print(f"\n=== Visualizing Vocab LoRA: {name} ===")
                    visualize_vocab_LoRA(module.lora_A_vocab, module.lora_B_vocab,original_W,label )

        

    global_e_max = config['stats']['global_energy_max']
    global_e_min = config['stats']['global_energy_min']
    stats = {"Initial_Energy_Max": global_e_max, "Initial_Energy_Min": global_e_min}
    
    # test_files = [twb,tab,tpb]  # for quick testing
    # test_files = [test_files[0],test_files[1],test_files[-2],test_files[-1]]  
    # test_files = test_files[:2] + test_files[-2:]  # for quick testing
    # test_files = test_files[:2] # for quick testing

    dataset = ECAL_Chunked_Dataset(test_files,max_seq_length=msl,
                energy_digitizer=energy_digitizer,verbose=True,
                ordering='energy',
                material_list=material_list,
                inference_mode=True)

    dataloader = CreateInferenceLoader(dataset,config)

    # choose compact dtypes safely
    token_dtype = np.uint16 if vocab_size <= 65535 else np.int32
    if digitize_energy:
        energy_dtype = np.uint16 if energy_vocab <= 65535 else np.int32
    else:
        energy_dtype = np.float32
        
    w = ShowerWriterCompound(outfile, token_dtype=token_dtype,
                            energy_dtype=energy_dtype, compression="lzf")

    kbar = pkbar.Kbar(target=len(dataloader), width=20)

    buffers = {"idx": [], "ene": [], "idx_t": [], "ene_t": [], "initE": [], "material_index": []}
    flush_size = config['Inference']['flush_size']

    for i, data in enumerate(dataloader):
        if i == args.num_showers:
            break

        pos, _ , initial_energy, material_index, initial_energy_t, ene = data
        pos = pos.to(device).long()
        initial_energy = initial_energy.to(device).float()
        material_index = material_index.to(device).long()
        initial_energy_t = initial_energy_t.numpy()

        torch.cuda.empty_cache()

        with torch.inference_mode():
            if args.use_amp:
                with autocast(dtype=torch.float16):     
                    generated_indices, generated_energies = model.generate(
                        initial_energy=initial_energy,
                        material_index=material_index,
                        method=args.sampling_method,
                        dynamic_temp=args.dynamic_temp,
                        max_seq_len=msl,
                        temperature=args.temperature,
                        use_kv_cache=args.use_kv_cache,particle_type=particle_type    
                    )
            else:
                generated_indices, generated_energies = model.generate(
                    initial_energy=initial_energy,
                    material_index=material_index,
                    method=args.sampling_method,
                    dynamic_temp=args.dynamic_temp,
                    max_seq_len=msl,
                    temperature=args.temperature,
                    use_kv_cache=args.use_kv_cache,particle_type=particle_type
                )

            generated_indices = generated_indices.detach().cpu().numpy()
            generated_energies = generated_energies.detach().cpu().numpy()

            true_indices = pos.detach().cpu().numpy()
            true_energies = ene.numpy().astype(np.float32)

            # collect into buffers (store each shower together)
            for b in range(generated_indices.shape[0]):
                # cast to chosen dtypes without copy when possible
                buffers["idx"].append(generated_indices[b].astype(token_dtype,  copy=False))
                buffers["idx_t"].append(true_indices[b].astype(token_dtype,  copy=False))
                if digitize_energy:
                    buffers["ene"].append(generated_energies[b].astype(energy_dtype, copy=False))
                    buffers["ene_t"].append(true_energies[b].astype(np.float32, copy=False))
                else:
                    buffers["ene_t"].append(true_energies[b].astype(np.float32, copy=False))
                    buffers["ene"].append(generated_energies[b].astype(np.float32, copy=False))

                buffers["initE"].append(float(initial_energy_t[b].item()))
                buffers['material_index'].append(int(material_index[b].item()))

                if len(buffers["idx"]) >= flush_size:
                    print(len(buffers["idx"]), "showers reached flush size. Writing to disk...")
                    w.append_block(buffers["idx"],
                                buffers["ene"],
                                buffers["idx_t"],
                                buffers["ene_t"],
                                buffers["initE"],
                                buffers["material_index"])
                    buffers = {"idx": [], "ene": [], "idx_t": [], "ene_t": [], "initE": [], "material_index": []}   

        # flush remainder
        if buffers["idx"]:
            w.append_block(buffers["idx"],
                        buffers["ene"],
                        buffers["idx_t"],
                        buffers["ene_t"],
                        buffers["initE"],
                        buffers["material_index"])
            buffers = {"idx": [], "ene": [], "idx_t": [], "ene_t": [], "initE": [], "material_index": []}

        kbar.update(i + 1)


    w.close()


    print("Generation complete. Output written to: ", outfile)
    print("Generating plots...")
    make_plots(outfile, energy_digitizer,materials_to_plot=materials_to_generate, num_showers=args.num_showers,material_list=material_list,comparison_path=args.comp_paths)


class ShowerWriterCompound:
    def __init__(self, path, token_dtype, energy_dtype,
                 compression="lzf", chunk_rows=1024):
        """
        Creates (or opens) /<method>/showers with dtype:
          initial_energy: float32
          indices:        vlen[token_dtype]
          energies:       vlen[energy_dtype]
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(path, "w", libver="latest")
        vlen_tok = h5py.vlen_dtype(token_dtype)
        vlen_eng = h5py.vlen_dtype(energy_dtype)
        vlen_float = h5py.vlen_dtype(np.float32)
        self.rec_dtype = np.dtype([
            ("initial_energy", np.float32),
            ("indices", vlen_tok),
            ("energies", vlen_eng),
            ("indices_true", vlen_tok),
            ("energies_true", vlen_float),
            ("material_index", np.int32),
        ])

        if "showers" not in self.f:
            self.dset = self.f.create_dataset(
                "showers", shape=(0,), maxshape=(None,), dtype=self.rec_dtype,
                chunks=(chunk_rows,), compression=compression, shuffle=False
            )
        else:
            self.dset = self.f["showers"]

    def append_block(self, indices_list, energies_list, indices_true_list, energies_true_list, initE_list, material_index_list):
        """
        indices_list / energies_list: list of 1D numpy arrays (same length within a shot)
        initE_list: list/array of float
        Appends a whole block at once (fast).
        """
        B = len(indices_list)
        block = np.empty(B, dtype=self.rec_dtype)
        block["initial_energy"] = np.asarray(initE_list, dtype=np.float32)
        # store vlen arrays; h5py accepts Python lists of np arrays
        block["indices"] = [np.asarray(t) for t in indices_list]
        block["energies"] = [np.asarray(e) for e in energies_list]
        block["indices_true"] = [np.asarray(t) for t in indices_true_list]
        block["energies_true"] = [np.asarray(e) for e in energies_true_list]
        block["material_index"] = np.asarray(material_index_list, dtype=np.int32)


        n0 = self.dset.shape[0]
        self.dset.resize((n0 + B,))
        self.dset[n0:n0+B] = block

    def close(self):
        self.f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate showers using trained GPT model.")
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: "cuda" or "cpu".')
    parser.add_argument('--sampling_method', type=str, default='Default', help='Sampling method. See model.generate() for options.')
    parser.add_argument('--use_kv_cache', action='store_true', help='Whether to use KV cache during generation.')
    parser.add_argument('--use_amp', action='store_true', help='Whether to use automatic mixed precision.')
    parser.add_argument('--num_showers', type=int, default=-1, help='Number of showers to generate for plotting. -1 for all.')
    parser.add_argument('--materials_to_generate', type=str, nargs='+', default=None, help='List of materials to generate showers for. If not set, generates for all materials in config.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature during generation.')
    parser.add_argument('--dynamic_temp', action='store_true', help='Whether to use dynamic temperature during generation.')
    parser.add_argument('--visualize_vocab_LoRA', action='store_true', help='Whether to visualize vocab LoRA matrices after loading model.')
    parser.add_argument('--comp_paths', type=str, nargs='+', default=None, help='Paths to additional models for comparison.')
    args = parser.parse_args()

    os.makedirs("Generations", exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config, args)