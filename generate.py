import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import h5py
from pathlib import Path
import pkbar
from datetime import datetime
import os
import argparse

from plotting import make_plots

from models.GPT import ECAL_GPT
from dataloader.tokenizer import EnergyTokenizer
from dataloader.dataset import ECAL_Chunked_Dataset
from dataloader.dataloader import CreateInferenceLoader
from utils.utils import read_text

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

    energy_res = config['stats']['token_energy_res']
    e_max = config['stats']['token_energy_max']
    e_min = config['stats']['token_energy_min']
   
    energy_digitizer = EnergyTokenizer(e_max=e_max, e_min=e_min, resolution=energy_res)
    outfile = os.path.join("Generations", config['Inference']['output_file'])

    if os.path.exists(outfile):
        print(f"Output file {outfile} already exists, do you want to overwrite it? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            print("Generation aborted by user, running plotting code only.")
            make_plots(outfile, energy_digitizer, material_list=material_list, num_showers=args.num_showers)
            exit(0)
        else:
            print("========= Generation Started =========")
            print("Device: ", args.device)
            print("Sampling Method: ", args.sampling_method)
            print("Using AMP: ", args.use_amp)
            print("Using KV Cache: ", args.use_kv_cache)
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
                        material_list=material_list).to(args.device)

            model_path = config['Inference']['model_path']
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["net_state_dict"])

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

            # (Optional) compile AFTER loading. If you stay on CPU, compiling may not help; feel free to skip.
            try:
                if use_kv_cache == True:
                    model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            except Exception as _:
                # torch.compile may not be available / useful on this env; continue without it
                print('Could not compile model. Continuing...')
                pass

            test_files = []
            for material in material_list:
                if "Pb" in material:
                    print("Adding Pb files to testing set.")
                    test_files += read_text(config['dataset']['testing']['Pb_test_files'])
                elif "W" in material:
                    print("Adding W files to testing set.")
                    test_files += read_text(config['dataset']['testing']['W_test_files'])
                elif "Ta" in material:
                    print("Adding Ta files to testing set.")
                    test_files += read_text(config['dataset']['testing']['Ta_test_files'])

            global_e_max = config['stats']['global_energy_max']
            global_e_min = config['stats']['global_energy_min']
            stats = {"Initial_Energy_Max": global_e_max, "Initial_Energy_Min": global_e_min}
            
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
                                max_seq_len=msl,
                                temperature=1.0,
                                use_kv_cache=args.use_kv_cache,    
                            )
                    else:
                        generated_indices, generated_energies = model.generate(
                            initial_energy=initial_energy,
                            material_index=material_index,
                            method=args.sampling_method,
                            max_seq_len=msl,
                            temperature=1.0,
                            use_kv_cache=args.use_kv_cache,
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
            make_plots(outfile, energy_digitizer, material_list=material_list, num_showers=args.num_showers)


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
    #parser.add_argument('--material', type=str, default='G4_W', help='Material to generate showers for [G4_W, G4_Pb, G4_Ta].')
    args = parser.parse_args()

    os.makedirs("Generations", exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config, args)