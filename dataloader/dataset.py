from torch.utils.data import Dataset
import numpy as np
import h5py
import os
import glob
from typing import Literal
import pickle
from tqdm import tqdm


class ECAL_Dataset(Dataset):

    def __init__(self, data_path,
                 max_seq_length=27000,
                 energy_digitizer=None,
                 in_memory: Literal['in_memory', 'per_file', 'as_needed'] = 'as_needed',
                 index_cache_path=None):

        self.index_cache_path = index_cache_path
        # Constant shape per shot
        self.shape = (30, 30, 30)
        self.max_seq_length = max_seq_length
        self.energy_digitizer = energy_digitizer

        self.file_paths = self._gather_file_paths(data_path)
        self.index_map = self._build_index()

        self.SOS_token = 0
        # Positional Tokens 1-27000
        self.EOS_token = 27000 + 1  # 27001
        self.pad_token = self.EOS_token + 1  # 27002

        # Energy tokens
        self.energy_EOS_token = 24938 + 1
        self.energy_pad_token = 24938 + 2

        self.in_memory = in_memory

        if self.in_memory == 'in_memory':
            self.memory_cache = self._load_all_into_memory()

        self.current_file = None
        self.current_data = {}  # maps local idx → (indices, values, energy)

        return

    def _load_all_into_memory(self):
        if self.energy_digitizer is None:
            raise ValueError("Energy digitizer must be provided for tokenization.")

        cache = []

        # Group by file
        file_to_groupkeys = {}
        for file_path, key in self.index_map:
            if file_path not in file_to_groupkeys:
                file_to_groupkeys[file_path] = []
            file_to_groupkeys[file_path].append(key)

        print('Loading Files Into Memory...')
        for file_path, keys in tqdm(file_to_groupkeys.items()):
            with h5py.File(file_path, "r") as f:
                for key in keys:
                    group = f[key]
                    indices = group["indices"][()]
                    values = group["values"][()]
                    initial_energy = group.attrs["initial_energy"].item()

                    # Tokenize and sort
                    tokens = self.energy_digitizer.tokenize((indices, values))
                    if self.max_seq_length < tokens.size:
                        topk_idx = np.argpartition(tokens, -self.max_seq_length)[-self.max_seq_length:]
                        sorted_positions = topk_idx[np.argsort(-tokens[topk_idx])]
                    else:
                        sorted_positions = np.argsort(-tokens)
                    sorted_energies = tokens[sorted_positions]

                    # Trim at first energy == 1
                    cut_index = np.argmax(sorted_energies == 1)
                    if sorted_energies[cut_index] != 1:
                        cut_index = len(sorted_energies)

                    sorted_positions = sorted_positions[:cut_index]
                    sorted_energies = sorted_energies[:cut_index]

                    # Add SOS/EOS
                    sorted_positions = np.insert(sorted_positions, 0, self.SOS_token)
                    sorted_positions = np.append(sorted_positions, self.EOS_token)

                    sorted_energies = np.insert(sorted_energies, 0, self.SOS_token)
                    sorted_energies = np.append(sorted_energies, self.energy_EOS_token)

                    cache.append((sorted_positions, sorted_energies, initial_energy))

        return cache

    def _gather_file_paths(self, data_path):
        if os.path.isdir(data_path):
            return sorted(glob.glob(os.path.join(data_path, "*.hdf5")))
        elif os.path.isfile(data_path):
            return [data_path]
        else:
            raise FileNotFoundError(f"No valid files found at {data_path}")

    def _build_index(self):
        if self.index_cache_path is not None and os.path.exists(self.index_cache_path):
            with open(self.index_cache_path, 'rb') as f:
                self.file_to_keys, index = pickle.load(f)
            print(f"[Cache] Loaded index from: {self.index_cache_path}")
            return index
        index = []
        self.file_to_keys = {}
        for file_path in self.file_paths:
            keys = []
            with h5py.File(file_path, "r") as f:
                for key in sorted(f.keys()):
                    group = f[key]
                    if "indices" in group and "values" in group and "initial_energy" in group.attrs:
                        keys.append(key)
            self.file_to_keys[file_path] = keys
            index.extend([(file_path, key) for key in keys])  # global idx: (file, index_in_file)
        print(f"Total valid samples indexed: {len(index)}")
        return index

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.in_memory == 'in_memory':
            return self.memory_cache[idx]
        elif self.in_memory == 'per_file':
            file_path, index = self.index_map[idx]

            # If the file is not loaded, load it
            if file_path != self.current_file:
                self.current_file = file_path
                self.current_data = {}  # Clear cache
                with h5py.File(file_path, "r") as f:
                    for key in self.file_to_keys[file_path]:
                        group = f[key]
                        indices = group["indices"][()]
                        values = group["values"][()]
                        energy = group.attrs["initial_energy"].item()
                        self.current_data[key] = (indices, values, energy)  # store by group name

            indices, values, initial_energy = self.current_data[index]

        elif self.in_memory == 'as_needed':
            file_path, key = self.index_map[idx]
            with h5py.File(file_path, "r") as f:
                group = f[key]
                indices = group["indices"][()]      # (N, 3)
                values = group["values"][()]        # (N,)
                initial_energy = group.attrs["initial_energy"].item()

        # Tokenization
        if self.energy_digitizer is None:
            raise ValueError("Energy digitizer must be provided for tokenization.")

        tokens = self.energy_digitizer.tokenize((indices, values))

        # Sort by energy (descending)
        sorted_positions = np.argsort(tokens)[::-1]
        sorted_energies = tokens[sorted_positions]

        # Cut sequence at first token with energy bin == 1
        cut_index = np.argmax(sorted_energies == 1)
        if sorted_energies[cut_index] != 1:
            cut_index = len(sorted_energies)

        sorted_positions = sorted_positions[:cut_index]
        sorted_energies = sorted_energies[:cut_index]

        # Add SOS/EOS tokens
        sorted_positions = np.insert(sorted_positions, 0, self.SOS_token)
        sorted_positions = np.append(sorted_positions, self.EOS_token)

        sorted_energies = np.insert(sorted_energies, 0, self.SOS_token)
        sorted_energies = np.append(sorted_energies, self.energy_EOS_token)

        return sorted_positions, sorted_energies, initial_energy
