from torch.utils.data import Dataset
import numpy as np
import h5py
import os
import glob


class ECAL_Dataset(Dataset):

    def __init__(self, data_path,
                 max_seq_length=27000,
                 energy_digitizer=None):

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

        return

    def _gather_file_paths(self, data_path):
        if os.path.isdir(data_path):
            return sorted(glob.glob(os.path.join(data_path, "*.hdf5")))
        elif os.path.isfile(data_path):
            return [data_path]
        else:
            raise FileNotFoundError(f"No valid files found at {data_path}")

    def _build_index(self):
        index = []
        for file_path in self.file_paths:
            with h5py.File(file_path, "r") as f:
                for key in f.keys():
                    group = f[key]
                    if "indices" in group and "values" in group and "initial_energy" in group.attrs:
                        index.append((file_path, key))
                    else:
                        print(f"Skipping: {file_path}, group '{key}' missing required fields.")

        print(f"Total valid samples indexed: {len(index)}")
        return index

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
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
