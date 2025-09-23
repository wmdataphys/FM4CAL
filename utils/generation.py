import torch
from torch.cuda.amp.grad_scaler import GradScaler
import numpy as np
import json
import h5py
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import os

from models.GPT import ECAL_GPT
from dataloader.tokenizer import EnergyTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("/sciclone/home/cjgranger/FM4CAL/Trained_Models/ecal_test___Sep-21-2025/config.json", "r") as f:
    config = json.load(f)

model_path = '/sciclone/home/cjgranger/FM4CAL/Trained_Models/ecal_test___Sep-21-2025/ecal_test_epoch02_val_loss_0.000531.pth'

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
use_amp = config['use_amp']
use_MoE = bool(config['model']['use_MoE'])

digitize_energy = True
print("Digitizing Energy - classification over adjacent vocabulary.")
print("Energy vocab: ", config['model']['energy_vocab'])
energy_res = config['stats']['energy_res']
e_max = config['stats']['energy_max']
e_min = config['stats']['energy_min']
print("E_Max: ", e_max, " E_Min: ", e_min, "E_Res: ", energy_res)
energy_digitizer = EnergyTokenizer(e_max=e_max, e_min=e_min, resolution=energy_res)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Running on CPU.")
else:
    print(f"Running on {torch.cuda.device_count()} GPU(s).")

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
            num_classes=num_classes,
            device=device).to(device)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["net_state_dict"])

scaler = None
if use_amp:
    scaler = GradScaler()
    if "scaler" in checkpoint:  # load scaler state if present
        scaler.load_state_dict(checkpoint["scaler"])
startEpoch = checkpoint.get("epoch", -1) + 1
history = checkpoint.get("history", {})
global_step = checkpoint.get("global_step", 0)
print(f"Loaded from: {model_path} at epoch {startEpoch}, global_step {global_step}")

model.eval()

# (Optional) compile AFTER loading. If you stay on CPU, compiling may not help; feel free to skip.
try:
    model = torch.compile(model, mode="reduce-overhead", dynamic=True)
except Exception as _:
    # torch.compile may not be available / useful on this env; continue without it
    print('Could not compile model. Continuing...')
    pass

energies = []
val_data = "/sciclone/data10/cjgranger/ECAL/ECAL_Cole/ECALSim/ILDConfig/StandardConfig/production/simulation/processed/val"
for file_path in os.listdir(val_data):
    with h5py.File(os.path.join(val_data, file_path), "r") as f:
        keys = f.keys()
        skipped = 0

        for key in keys:
            group = f[key]
            initial_energy = group.attrs['initial_energy'][0]
            energies.append(initial_energy)
# convert list -> tensor on the right device
energies = torch.tensor(energies, dtype=torch.float32, device=device)
# sampling_methods = ["Nucleus", "TopK", "Default", "Greedy", "Min_p"]
sampling_methods = ["Default"]


class ShotWriterCompound:
    def __init__(self, path, method, token_dtype, energy_dtype,
                 compression="lzf", chunk_rows=1024):
        """
        Creates (or opens) /<method>/shots with dtype:
          initial_energy: float32
          indices:        vlen[token_dtype]
          energies:       vlen[energy_dtype]
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(path, "a", libver="latest")
        g = self.f.require_group(str(method))

        vlen_tok = h5py.vlen_dtype(token_dtype)
        vlen_eng = h5py.vlen_dtype(energy_dtype)
        self.rec_dtype = np.dtype([
            ("initial_energy", np.float32),
            ("indices", vlen_tok),
            ("energies", vlen_eng),
        ])

        if "shots" not in g:
            self.dset = g.create_dataset(
                "shots", shape=(0,), maxshape=(None,), dtype=self.rec_dtype,
                chunks=(chunk_rows,), compression=compression, shuffle=True
            )
        else:
            self.dset = g["shots"]

    def append_block(self, indices_list, energies_list, initE_list):
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

        n0 = self.dset.shape[0]
        self.dset.resize((n0 + B,))
        self.dset[n0:n0+B] = block

    def close(self):
        self.f.close()


def energy_key(val: float, ndigits: int = 6) -> str:
    return f"{float(val):.{ndigits}f}"


# --- replace your save_generated(...) with this batched version ---
@torch.inference_mode()
def save_generated(outfile, model, sampling_methods, energies,
                   batch_size=64, max_seq_len=1700, flush_size=1024):
    print(f"Writing generated samples to: {outfile}")

    # choose compact dtypes safely
    token_dtype = np.uint16 if vocab_size <= 65535 else np.int32
    if digitize_energy:
        energy_dtype = np.uint16 if energy_vocab <= 65535 else np.int32
    else:
        energy_dtype = np.float32

    # one writer per sampling method
    writers = {
        m: ShotWriterCompound(outfile, m, token_dtype=token_dtype,
                              energy_dtype=energy_dtype, compression="lzf")
        for m in sampling_methods
    }
    # RAM buffers per method (flush in big blocks)
    buffers = {m: {"idx": [], "ene": [], "initE": []} for m in sampling_methods}

    nE = energies.numel()
    for method in sampling_methods:
        w = writers[method]
        for start in tqdm(range(0, nE, batch_size), desc=method):
            stop = min(start + batch_size, nE)
            batch = energies[start:stop].reshape(-1, 1)  # (B,1) on device

            # generate one shot per initial_energy
            idx_batch, e_batch = model.generate(
                initial_energy=batch, method=method, max_seq_len=max_seq_len
            )

            # move to CPU
            idx_batch = idx_batch.detach().cpu().numpy()
            e_batch = e_batch.detach().cpu().numpy()

            # collect into buffers (store each shot together)
            for b in range(idx_batch.shape[0]):
                # cast to chosen dtypes without copy when possible
                buffers[method]["idx"].append(idx_batch[b].astype(token_dtype,  copy=False))
                if digitize_energy:
                    buffers[method]["ene"].append(e_batch[b].astype(energy_dtype, copy=False))
                else:
                    buffers[method]["ene"].append(e_batch[b].astype(np.float32, copy=False))
                buffers[method]["initE"].append(float(batch[b, 0].item()))

                if len(buffers[method]["idx"]) >= flush_size:
                    w.append_block(buffers[method]["idx"],
                                   buffers[method]["ene"],
                                   buffers[method]["initE"])
                    buffers[method] = {"idx": [], "ene": [], "initE": []}

        # flush remainder
        if buffers[method]["idx"]:
            w.append_block(buffers[method]["idx"],
                           buffers[method]["ene"],
                           buffers[method]["initE"])
            buffers[method] = {"idx": [], "ene": [], "initE": []}

    for w in writers.values():
        w.close()




# ------- NEW: build date- and model-based filename -------
run_date = datetime.now().strftime("%b-%d-%Y")  # e.g., 'Sep-02-2025'
model_name = Path(model_path).stem              # e.g., 'ecal_test_epoch99_val_loss_8.982370'
base_dir = Path("/sciclone/data10/cjgranger/ECALGPT_generated")

outfile = base_dir / run_date / f"{model_name}.hdf5"
# --------------------------------------------------------
save_generated(outfile, model, sampling_methods, energies)
