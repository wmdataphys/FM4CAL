import os

# Set these BEFORE importing torch
os.environ["TORCH_LOGS"] = "-cudagraphs"
os.environ["GLOG_minloglevel"] = "3"

import torch
from torch.cuda.amp import autocast
import numpy as np
import json
import h5py
from pathlib import Path
import pkbar
import argparse
import warnings
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

warnings.filterwarnings("ignore", message=".*weights_only.*")

from plotting import make_plots, visualize_vocab_LoRA
from models.GPT import ECAL_GPT
from dataloader.tokenizer import EnergyTokenizer
from dataloader.dataset import ECAL_Chunked_Dataset
from dataloader.dataloader import CreateDistInferenceLoader
from utils.utils import read_text, singular_value_checks


def create_model(config,expanded_seq_len=None):
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
    
    if expanded_seq_len is not None:
        msl = expanded_seq_len
        is_expanded = True
    else:
        is_expanded = False
    
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
                material_list=material_list,
                base_model_type=base_model_type,
                particle_list=particle_list,
                LoRA_r=loRA_r,
                LoRA_alpha=loRA_alpha,
                enable_head_LoRA=enable_head_LoRA,
                enable_vocab_LoRA=enable_vocab_LoRA,
                enable_embedding_adapter=enable_embedding_adapter,
                vocab_LoRA_scale=vocab_LoRA_scale,
                use_RoPE=use_RoPE, is_expanded=is_expanded
                )

    return net 

class Generator:
    def __init__(self, config, rank, world_size, model, args):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.device = torch.device(f"cuda:{rank}")
        self.model = model.to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])
        self.args = args
        self.stats = config['stats']
        self.max_seq_length = config['model']['max_seq_length'] if args.gen_seq_len is None else args.gen_seq_len
        self.material_list = config['material_list']
        self.digitize_energy = config['digitize_energy']
        self.energy_vocab = config['model']['energy_vocab']
        self.vocab_size = config['model']['vocab_size']
        if self.digitize_energy:
            token_energy_res = config['stats']['token_energy_res']
            token_e_max = config['stats']['token_energy_max']
            token_e_min = config['stats']['token_energy_min'] 
            self.energy_digitizer = EnergyTokenizer(e_max=token_e_max, e_min=token_e_min, resolution=token_energy_res)

            if self.rank == 0:
                print("Digitizing Energy.")
                print("Energy vocab: ", config['model']['energy_vocab'])
                print("Token_E_Max: ", token_e_max, " E_Min: ", token_e_min, "E_Res: ", token_energy_res)
        else:
            self.energy_digitizer = None
            if self.rank == 0:
                print("Regression over energy. No digitization applied.")

        if self.rank == 0:
            print("========= Generation Started =========")
            print("Device: ", self.args.device)
            print("Using AMP: ", self.args.use_amp)
            print("Using KV Cache: ", self.args.use_kv_cache)
            print("Sampling method: ", self.args.sampling_method)
            print("Temperature: ", self.args.temperature) if not self.args.dynamic_temp else print("Dynamic Temperature: Enabled")
            print("Generating showers for materials: ", self.args.materials_to_generate)
            print("Number of showers to generate: ", self.args.num_showers)
            print("Maximum sequence length for generation: ", self.max_seq_length)
            print("Maximum model sequence length: ", args.expanded_seq_len if args.gen_seq_len is not None else config['model']['max_seq_length'])
            print("=====================================")

        # choose compact dtypes safely
        self.token_dtype = np.uint16 if self.vocab_size <= 65535 else np.int32
        if self.digitize_energy:
            self.energy_dtype = np.uint16 if self.energy_vocab <= 65535 else np.int32
        else:
            self.energy_dtype = np.float32

        # Will remove this crime later - k8s giving permission errors and don't feel like debugging right now
        outfile = os.path.join("/sciclone/scr30/jgiroux/FM4CAL/Generations", self.args.output_file if self.args.output_file is not None else self.config['Inference']['output_file'])
        outfile = outfile.replace('.h5', f'_rank{self.rank}.h5')
        self.w = ShowerWriterCompound(outfile, token_dtype=self.token_dtype,
                            energy_dtype=self.energy_dtype, compression="lzf")

    def init_kbar(self, num_files, num_epochs=1):
        total_samples = num_files * self.config['dataset']['tracks_per_file'] # 10k per file roughly - overestimation
        total_samples = total_samples // self.world_size
        per_gpu_bs = self.config['Inference']['batch_size'] 
        total_batches = math.ceil(total_samples / per_gpu_bs)
        self.kbar = pkbar.Kbar(target=total_batches, width=20)
            
    def load_chunked_dataset(self,file_list,verbose=False):
        global_e_max = self.stats['global_energy_max']
        global_e_min = self.stats['global_energy_min']
        stats = {"Initial_Energy_Max": global_e_max, "Initial_Energy_Min": global_e_min}
        dataset = ECAL_Chunked_Dataset(file_list=file_list,max_seq_length=self.max_seq_length,energy_digitizer=self.energy_digitizer
                                       ,verbose=verbose,ordering='energy',global_stats=stats,
                                       material_list=self.material_list, inference_mode=True, n_events=self.args.num_showers)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        loader = CreateDistInferenceLoader(dataset,sampler,config=self.config)
        return loader, sampler

    def load_weights(self, checkpoint):
        try:
            self.model.load_state_dict(checkpoint["net_state_dict"],strict=True)
        except Exception as e:
            print(f"Error loading state dict: {e} on rank {self.rank}")
            print("Did you forget to add a material or particle to the config file?")
            exit(1)

        if self.rank == 0:
            startEpoch = checkpoint.get("epoch", -1) + 1
            history = checkpoint.get("history", {})
            global_step = checkpoint.get("global_step", 0)
            print(f"Loaded model at epoch {startEpoch}, global_step {global_step}")

    def generate_data(self, loader, flush_size: int = int(1e5), particle_type="gamma"):
        self.model.eval()

        try:
            if self.args.use_kv_cache == True:
                self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=True)
        except Exception as _:
            # torch.compile may not be available / useful on this env; continue without it
            print('Could not compile model. Continuing...')
            pass

        buffers = {"idx": [], "ene": [], "idx_t": [], "ene_t": [], "initE": [], "material_index": []}

        for i, data in enumerate(loader):

            pos, _ , initial_energy, material_index, initial_energy_t, ene = data
            pos = pos.to(self.device).long()
            initial_energy = initial_energy.to(self.device).float()
            material_index = material_index.to(self.device).long()
            initial_energy_t = initial_energy_t.numpy()

            torch.cuda.empty_cache()

            with torch.inference_mode():
                if self.args.use_amp:
                    with autocast(dtype=torch.float16):     
                        generated_indices, generated_energies = self.model.module.generate(
                            initial_energy=initial_energy,
                            material_index=material_index,
                            method=self.args.sampling_method,
                            dynamic_temp=self.args.dynamic_temp,
                            max_seq_len=self.max_seq_length,
                            temperature=self.args.temperature,
                            use_kv_cache=self.args.use_kv_cache,particle_type=particle_type    
                        )
                else:
                    generated_indices, generated_energies = self.model.module.generate(
                        initial_energy=initial_energy,
                        material_index=material_index,
                        method=self.args.sampling_method,
                        dynamic_temp=self.args.dynamic_temp,
                        max_seq_len=self.max_seq_length,
                        temperature=self.args.temperature,
                        use_kv_cache=self.args.use_kv_cache,particle_type=particle_type
                    )

                generated_indices = generated_indices.detach().cpu().numpy()
                generated_energies = generated_energies.detach().cpu().numpy()

                true_indices = pos.detach().cpu().numpy()
                true_energies = ene.numpy().astype(np.float32)

                # collect into buffers (store each shower together)
                for b in range(generated_indices.shape[0]):
                    # cast to chosen dtypes without copy when possible
                    buffers["idx"].append(generated_indices[b].astype(self.token_dtype,  copy=False))
                    buffers["idx_t"].append(true_indices[b].astype(self.token_dtype,  copy=False))
                    if self.digitize_energy:
                        buffers["ene"].append(generated_energies[b].astype(self.energy_dtype, copy=False))
                        buffers["ene_t"].append(true_energies[b].astype(np.float32, copy=False))
                    else:
                        buffers["ene_t"].append(true_energies[b].astype(np.float32, copy=False))
                        buffers["ene"].append(generated_energies[b].astype(np.float32, copy=False))

                    buffers["initE"].append(float(initial_energy_t[b].item()))
                    buffers['material_index'].append(int(material_index[b].item()))

                    if len(buffers["idx"]) >= flush_size:
                        print(len(buffers["idx"]), "showers reached flush size. Writing to disk...")
                        self.w.append_block(buffers["idx"],
                                    buffers["ene"],
                                    buffers["idx_t"],
                                    buffers["ene_t"],
                                    buffers["initE"],
                                    buffers["material_index"])
                        buffers = {"idx": [], "ene": [], "idx_t": [], "ene_t": [], "initE": [], "material_index": []}   

            # flush remainder
            if buffers["idx"]:
                self.w.append_block(buffers["idx"],
                            buffers["ene"],
                            buffers["idx_t"],
                            buffers["ene_t"],
                            buffers["initE"],
                            buffers["material_index"])
                buffers = {"idx": [], "ene": [], "idx_t": [], "ene_t": [], "initE": [], "material_index": []}

            self.kbar.update(i + 1)

        self.w.close()

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


def run_worker(rank, world_size, config, file_list, particle_type, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model_path = config['Inference']['model_path'] if args.model_path is None else args.model_path
    checkpoint = torch.load(model_path, map_location=torch.device(f'cuda:{rank}'))   

    model = create_model(config, expanded_seq_len=args.expanded_seq_len)
    try:
        model.load_state_dict(checkpoint["net_state_dict"],strict=True)
    except Exception as e:
        print(f"Error loading state dict: {e} on rank {rank}")
        print("Did you forget to add a material or particle to the config file?")
        exit(1)

    if rank == 0:
        startEpoch = checkpoint.get("epoch", -1) + 1
        history = checkpoint.get("history", {})
        global_step = checkpoint.get("global_step", 0)
        print(f"Loaded model at epoch {startEpoch}, global_step {global_step}")

        for name, param in model.named_parameters():
            if "lpe_expansion" in name:
                print(f"LPE Expansion Parameter {name} has length: {param.shape} and value: {param.mean().data.cpu().numpy()}")
                lpe_expansion_weight = param.data
                if "pos" in name and not "energy" in name:
                    orig_weight = model.pos_embedding.weight.data
                elif "energy" in name:
                    orig_weight = model.energy_pos_embedding.weight.data
                else:
                    print("Unknown LPE expansion type in name: ", name)
                    continue

                check_ = torch.allclose(lpe_expansion_weight[:orig_weight.shape[0]], orig_weight, atol=1e-5)
                print(f"Check if LPE expansion weight matches original positional embedding for first {orig_weight.shape[0]} tokens: {check_}")
                

    Generator_instance = Generator(config, rank, world_size, model, args)
    flush_size = config['Inference']['flush_size']
    if rank == 0: print(f"Starting generation on rank {rank}...")
    Generator_instance.init_kbar(num_files=len(file_list))
    loader, sampler = Generator_instance.load_chunked_dataset(file_list,verbose=(rank==0))
    Generator_instance.generate_data(loader, flush_size=flush_size, particle_type=particle_type)
    if rank == 0: print(f"Generation completed on rank {rank}.")

    dist.barrier()
    dist.destroy_process_group()


def merge_hdf5_files(base_filename, world_size):
    rank0_file = base_filename.replace('.h5', '_rank0.h5')
    with h5py.File(rank0_file, "r") as f:
        rec_dtype = f["showers"].dtype

    with h5py.File(base_filename, "w", libver="latest") as out_f:
        dset = out_f.create_dataset(
            "showers", shape=(0,), maxshape=(None,), dtype=rec_dtype,
            chunks=(1024,), compression="lzf", shuffle=False
        )
        
        for rank in range(world_size):
            rank_file = base_filename.replace('.h5', f'_rank{rank}.h5')
            if os.path.exists(rank_file):
                with h5py.File(rank_file, "r") as rank_f:
                    rank_data = rank_f["showers"][:]
                    n0 = dset.shape[0]
                    dset.resize((n0 + len(rank_data),))
                    dset[n0:] = rank_data
                
                os.remove(rank_file)
                print(f"Merged rank {rank} file")
    
    print(f"Final merged file: {base_filename}")


def read_text(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        lines = [line.strip() for line in lines]
        return lines

    except FileNotFoundError:
        raise ValueError(f"Error: The file '{file_path}' was not found.")

def main(config,args):

    material_list = args.material_list if args.material_list is not None else config['material_list']
    particle_list = args.particle_list if args.particle_list is not None else config['particle_list']

    # Replace in config for downstream use (e.g., in Generator class)
    config['material_list'] = material_list
    config['particle_list'] = particle_list

    if args.materials_to_generate is not None:
        materials_to_generate = args.materials_to_generate
        print("Generating for specified materials: ", materials_to_generate)
    else:
        materials_to_generate = material_list
        print("Generating for all materials in config: ", materials_to_generate)

    outfile = os.path.join("/sciclone/scr30/jgiroux/FM4CAL/Generations", args.output_file if args.output_file is not None else config['Inference']['output_file'])

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Skip this for K8 deployment
    # if local_rank == 0:
    #     if os.path.exists(outfile) and not args.inference_only:
    #         print(f"Output file {outfile} already exists, do you want to overwrite it? (y/n)")
    #         response = input().strip().lower()
    #         if response != 'y':
    #             energy_digitizer = EnergyTokenizer(e_max=config['stats']['token_energy_max'], e_min=config['stats']['token_energy_min'], resolution=config['stats']['token_energy_res'])
    #             print("Generation aborted by user, running plotting code only.")
    #             make_plots(outfile, energy_digitizer,materials_to_plot=materials_to_generate, num_showers=args.num_showers,material_list=material_list,comparison_path=args.comp_paths)
    #             exit(0)


    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

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

    # test_files = test_files[:2]

    run_worker(rank, world_size, config, test_files, particle_type, args)

    if local_rank == 0:
        print("Merging rank-specific files...")
        merge_hdf5_files(outfile, world_size)
        if not args.inference_only:
            print("Generating plots...")
            energy_digitizer = EnergyTokenizer(e_max=config['stats']['token_energy_max'], e_min=config['stats']['token_energy_min'], resolution=config['stats']['token_energy_res'])
            make_plots(outfile, energy_digitizer, materials_to_plot=materials_to_generate, 
                      num_showers=args.num_showers, material_list=material_list,
                      comparison_path=args.comp_paths)

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
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint for generation. Overrides config if set.')
    parser.add_argument('--output_file', type=str, default=None, help='Output file name for generated showers. Overrides config if set.')
    parser.add_argument('--inference_only', action='store_true', help='If set, only runs inference and plotting without generation. Assumes output file already exists.')
    parser.add_argument('--particle_list', type=str, nargs='+', default=None, help='List of particles to use. Overrides config if set.')
    parser.add_argument('--material_list', type=str, nargs='+', default=None, help='List of materials to use. Overrides config if set.')
    parser.add_argument('--expanded_seq_len', type=int, default=None, help='If set, expands the model sequence length to this value. Overrides config if set.')
    parser.add_argument('--gen_seq_len', type=int, default=None, help='Maximum sequence length for generation. Overrides config if set.')
    args = parser.parse_args()

    # os.makedirs("Generations", exist_ok=True)

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config, args)
