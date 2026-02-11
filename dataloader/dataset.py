import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
import glob
from typing import Literal
import pickle
from tqdm import tqdm
import time
import random
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist

class ECAL_Chunked_Dataset(Dataset):
    def __init__(self, file_list,
                 max_seq_length=2100,
                 energy_digitizer=None,verbose=False,
                 ordering: Literal['energy', 'spatial'] = 'energy',
                 material_list: list = ["G4_W_gamma", "G4_Ta_gamma"],
                 global_stats: dict = {"Initial_Energy_Max": 100.0, "Initial_Energy_Min": 10.0},
                 inference_mode: bool = False, n_events = None,dataset_seed=42
                    ):

        # Constant shape per shower
        self.shape = (30, 30, 30)
        self.max_seq_length = max_seq_length
        self.energy_digitizer = energy_digitizer
        self.verbose = verbose
        self.ordering = ordering
        self.material_list = material_list
        self.SOS_token = 0
        # Positional Tokens 1-27000
        self.EOS_token = 27000 + 1  # 27001
        self.pad_token = self.EOS_token + 1  # 27002
        self.global_stats = global_stats
        self.inference_mode = inference_mode
        self.n_events = n_events
        self.dataset_seed = dataset_seed
        self.rng = random.Random(self.dataset_seed)

        # Energy tokens
        self.energy_EOS_token = 25000 + 1
        self.energy_pad_token = 25000 + 2

        self.file_list = file_list
        if self.inference_mode:
            self.memory_cache, self.ground_truth_cache = self._load_all_into_memory()
        else:
            self.memory_cache = self._load_all_into_memory()

    def _encode_sample(self, indices, values):
        flat = np.ravel_multi_index(indices.T, self.shape) + 1  # shift by 1 to match token indexing, SOS = 0
        mask = (values > self.energy_digitizer.e_min + self.energy_digitizer.energy_res) & (values < self.energy_digitizer.e_max - self.energy_digitizer.energy_res)
        pixels = flat[mask]
        energies = values[mask]
        energy_copy = None

        if self.inference_mode:
            energy_copy = energies.copy()

        energies = self.energy_digitizer.tokenize(energies)  # Tokenize energies only for nonzero voxels

        if self.ordering == 'energy':
            order = np.argsort(energies)[::-1]  # descending
            pixels_sorted = pixels[order]
            energies_sorted = energies[order]
            if self.inference_mode:
                energy_copy = energy_copy[order]

        elif self.ordering == 'spatial':  # spatial
            order = np.argsort(pixels)  # ascending
            pixels_sorted = pixels[order]
            energies_sorted = energies[order]
            if self.inference_mode:
                energy_copy = energy_copy[order]
        else:
            raise ValueError(f"Unknown ordering: {self.ordering}")

        assert(len(pixels_sorted) == len(energies_sorted))

        return pixels_sorted, energies_sorted, energy_copy 

    def _load_all_into_memory(self):
        if self.energy_digitizer is None:
            raise ValueError("Energy digitizer must be provided for tokenization.")

        cache = []
        file_to_groupkeys = {}
        ground_truth_cache = []

        if self.verbose:
            print('Loading Files Into Memory...')

        for file_path in self.file_list:
            if self.verbose:
                print(f'Processing file: {file_path}')
            with h5py.File(file_path, "r") as f:
                for key in f.keys():
                    group = f[key]
                    indices = group["indices"][()]

                    if len(indices) < 1:
                        continue

                    values = group["values"][()]
                    initial_energy = group.attrs["initial_energy"].item()
                    material = group['material'][()].decode('utf-8')

                    if initial_energy < self.global_stats["Initial_Energy_Min"] or initial_energy > self.global_stats["Initial_Energy_Max"]:
                        continue

                    material_index = self.material_list.index(material)
                    initial_energy = ((initial_energy - self.global_stats["Initial_Energy_Min"]) / (self.global_stats["Initial_Energy_Max"] - self.global_stats["Initial_Energy_Min"])) * 2 - 1.0 # Scale to roughly -1 to 1

                    # Tokenize + order
                    sorted_positions, sorted_energies, energy_copy = self._encode_sample(indices, values)

                    cache.append((sorted_positions, sorted_energies, initial_energy, material_index))

                    if self.inference_mode:
                        # We will decode the pixels later, but use ground truth energies for plotting   
                        ground_truth_cache.append((initial_energy, energy_copy))


        if not self.inference_mode:
            if self.n_events is not None:
                idx = list(range(len(cache)))
                self.rng.shuffle(idx)
                cache = [cache[i] for i in idx[:self.n_events]]
            else:
                random.shuffle(cache)
                
            return cache

        else:
            return cache,ground_truth_cache

    def __len__(self):
        return len(self.memory_cache)

    def __getitem__(self, idx):
        pos, ene, initial_energy, material_index = self.memory_cache[idx]
        
        assert len(pos) == len(ene)
        
        # Truncate if too long (reserve space for SOS/EOS)
        if len(pos) > self.max_seq_length - 2:
            pos = pos[:self.max_seq_length - 2]  # ← Reserve 2 spaces
            ene = ene[:self.max_seq_length - 2]
        
        # Add SOS/EOS
        pos = np.insert(pos, 0, self.SOS_token)
        ene = np.insert(ene, 0, self.SOS_token)
        pos = np.append(pos, self.EOS_token)
        ene = np.append(ene, self.energy_EOS_token)
        
        # Pad if necessary
        if len(pos) < self.max_seq_length:
            pos = np.pad(pos, (0, self.max_seq_length - len(pos)), constant_values=self.pad_token)
            ene = np.pad(ene, (0, self.max_seq_length - len(ene)), constant_values=self.energy_pad_token)


        if not self.inference_mode:
            return pos, ene, initial_energy, material_index

        else:
            initial_energy_t, energy_copy = self.ground_truth_cache[idx]
            

            if len(energy_copy) > self.max_seq_length - 2:
                energy_copy = energy_copy[:self.max_seq_length - 2]  # Mimick reserving two spaces, apply -1 for SOS/EOS

            energy_copy = np.insert(energy_copy, 0, -1.0)  # SOS energy = -1
            energy_copy = np.append(energy_copy, -1.0)  # EOS energy = -1

            if len(energy_copy) < self.max_seq_length:
                # Have to pad for collate function
                energy_copy = np.pad(energy_copy, (0, self.max_seq_length - len(energy_copy)), constant_values=-1.0)  # Pad energy copy with -1

            return pos, ene, initial_energy, material_index, initial_energy_t, energy_copy

