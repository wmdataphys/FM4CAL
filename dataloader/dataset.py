import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
import glob
from typing import Literal
import pickle
from tqdm import tqdm
import random
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist

class ECAL_Chunked_Dataset(Dataset):
    def __init__(self, file_list,
                 max_seq_length=1700,
                 energy_digitizer=None,verbose=False,
                 ordering: Literal['energy', 'spatial'] = 'energy',
                 material_list: list = ["G4_W", "G4_Ta"],
                 global_stats: dict = {"Initial_Energy_Max": 100.0, "Initial_Energy_Min": 10.0},
                 inference_mode: bool = False
                    ):

        # Constant shape per shot
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

        energies = self.energy_digitizer.tokenize(energies)  # Tokenize energies only for nonzero voxels

        if self.ordering == 'energy':
            order = np.argsort(energies)[::-1]  # descending
            pixels_sorted = pixels[order]
            energies_sorted = energies[order]

        elif self.ordering == 'spatial':  # spatial
            order = np.argsort(pixels)  # ascending
            pixels_sorted = pixels[order]
            energies_sorted = energies[order]
        else:
            raise ValueError(f"Unknown ordering: {self.ordering}")

        assert(len(pixels_sorted) == len(energies_sorted))

        return pixels_sorted, energies_sorted

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
                    values = group["values"][()]
                    initial_energy = group.attrs["initial_energy"].item()
                    material = group['material'][()].decode('utf-8')

                    if initial_energy < self.global_stats["Initial_Energy_Min"] or initial_energy > self.global_stats["Initial_Energy_Max"]:
                        continue

                    if self.inference_mode:
                        ground_truth_cache.append(initial_energy)

                    material_index = self.material_list.index(material)
                    initial_energy = ((initial_energy - self.global_stats["Initial_Energy_Min"]) / (self.global_stats["Initial_Energy_Max"] - self.global_stats["Initial_Energy_Min"])) * 2 - 1.0 # Scale to roughly -1 to 1

                    # Tokenize + order
                    sorted_positions, sorted_energies = self._encode_sample(indices, values)

                    # # Add SOS/EOS
                    # sorted_positions = np.insert(sorted_positions, 0, self.SOS_token)
                    # sorted_positions = np.append(sorted_positions, self.EOS_token)

                    # sorted_energies = np.insert(sorted_energies, 0, self.SOS_token)
                    # sorted_energies = np.append(sorted_energies, self.energy_EOS_token)

                    cache.append((sorted_positions, sorted_energies, initial_energy, material_index))


        if not self.inference_mode:
            random.shuffle(cache)
            return cache

        else:
            return cache,ground_truth_cache

    def __len__(self):
        return len(self.memory_cache)

    def __getitem__(self, idx):
        pos, ene, initial_energy, material_index = self.memory_cache[idx]
        initial_energy_t = self.ground_truth_cache[idx]
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
            return pos, ene, initial_energy, material_index, initial_energy_t

