import numpy as np
import h5py
import torch
import torch.nn as nn

def read_text(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        lines = [line.strip() for line in lines]
        return lines

    except FileNotFoundError:
        raise ValueError(f"Error: The file '{file_path}' was not found.")


def energy_loss_fn(true_energies, pred_energies, padding_mask):
    """
    Computes SmoothL1Loss for energy regression from ECAL simulated data.

    Parameters:
    - true_energies: Tensor of shape (B, T)
    - pred_energies: Tensor of shape (B, T, 1) or (B, T)
    - padding_mask: Bool tensor of shape (B, T) where True = pad (ignored)

    Returns:
    - Scalar loss (float)
    """

    # Make sure shapes are compatible
    true_energies = true_energies.view(-1).float()
    pred_energies = pred_energies.view(-1).float()

    # Convert padding mask to 0 for pad, 1 for valid
    valid_mask = ~padding_mask.view(-1)

    # Compute unreduced SmoothL1 loss
    loss = nn.SmoothL1Loss(reduction='none')(pred_energies, true_energies)

    # Mask out padded positions
    loss = loss * valid_mask

    # Average only over non-padded elements
    return loss.sum() / valid_mask.sum()


def map_3d_to_1d(i, j, k, shape=(30, 30, 30)):
    x, y, z = shape
    return i * y * z + j * z + k


def map_1d_to_3d(index, shape=(30, 30, 30)):
    x, y, z = shape
    i = index // (y * z)
    rem = index % (y * z)
    j = rem // z
    k = rem % z
    return i, j, k


def compress_hdf5(filename, new_filename):
    with h5py.File(filename, "r") as f:
        group = f['30x30']
        # Move data into numpy arr
        energy = group['energy'][()]  # Incident Energy
        layers = group['layers'][()]  # 3D pixel reconstruciton

    with h5py.File(new_filename, "w") as f:
        for i in range(layers.shape[0]):
            sample = layers[i]
            indices = np.argwhere(sample != 0)
            values = sample[sample != 0]
            grp = f.create_group(f"{i}")
            grp.create_dataset("indices", data=indices, compression="gzip")
            grp.create_dataset("values", data=values, compression="gzip")
            grp.attrs["shape"] = sample.shape
            grp.attrs['initial_energy'] = energy[i]


def decompress_hdf5_to_dense(filename):
    with h5py.File(filename, "r") as f:
        num_samples = len(f.keys())

        # Initialize empty arrays
        energy = np.zeros((num_samples, 1), dtype=np.float32)
        # Adjust shape if needed
        layers = np.zeros((num_samples, 30, 30, 30), dtype=np.float32)

        for i in range(num_samples):
            grp = f[str(i)]
            indices = grp["indices"][:]
            values = grp["values"][:]
            shape = grp.attrs["shape"]

            # Reconstruct sparse sample to dense
            dense_sample = np.zeros(shape, dtype=values.dtype)
            for idx, val in zip(indices, values):
                dense_sample[tuple(idx)] = val

            layers[i] = dense_sample
            energy[i] = grp.attrs["initial_energy"]

    return energy, layers


def sparse_to_spatial_lists(indices, values, energy_tokenizer,
                            shape=(30, 30, 30),
                            drop_background=False,
                            background_bin=1):
    """
    Convert sparse (indices, values) → two 1D arrays:
      - positions: flat indices in top-left → bottom-right order per layer
      - energy_tokens: digitized energies in the same order

    indices: (N, 3) int, assumed (z, y, x)
    values:  (N,)   float
    """

    # 1. Flatten positions into raster order index
    flat = np.ravel_multi_index(indices.T, shape)   # (N,)

    # 2. Sort by flat index → layer, row, col order
    order = np.argsort(flat)
    flat_sorted = flat[order]
    vals_sorted = values[order]

    # 3. Digitize energies using the tokenizer's bins
    energy_tokens = np.digitize(vals_sorted, energy_tokenizer.e_bins)

    # 4. Optional: drop background (e.g., token == 1)
    if drop_background:
        mask = (energy_tokens != background_bin)
        flat_sorted = flat_sorted[mask]
        energy_tokens = energy_tokens[mask]

    return flat_sorted, energy_tokens


def log_vram_usage(tag=""):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"VRAM_LOG,{tag},{allocated:.2f},{reserved:.2f},{peak_allocated:.2f}")
    return allocated, reserved, peak_allocated
