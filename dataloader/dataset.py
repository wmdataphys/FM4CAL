from torch.utils.data import Dataset
import numpy as np
import h5py


class ECAL_Dataset(Dataset):

    def __init__(self, data_path,
                 max_seq_length=27000,
                 energy_digitizer=None):

        # Constant shape per shot
        self.shape = (30, 30, 30)
        self.file = h5py.File(data_path, "r")
        # Each key is an event index (0, 1 ...)
        self.keys = list(self.file.keys())
        self.max_seq_length = max_seq_length
        self.energy_digitizer = energy_digitizer

        self.SOS_token = 0
        # Positional Tokens 1-27000
        self.EOS_token = max_seq_length + 1  # 27001
        self.pad_token = self.EOS_token + 1  # 27002

        # Energy tokens
        self.energy_EOS_token = len(energy_digitizer.e_bins) + 1
        self.energy_pad_token = len(energy_digitizer.e_bins) + 2

        return

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        group = self.file[key]

        initial_energy = group.attrs["initial_energy"]
        indices = group["indices"][()]  # (N, 3)
        values = group["values"][()]  # (N,)

        if self.energy_digitizer is not None:
            tokens = self.energy_digitizer.tokenize((indices, values))

        sorted_positions = np.argsort(tokens)[::-1]
        sorted_energies = tokens[sorted_positions]

        sorted_positions = np.insert(sorted_positions, 0, self.SOS_token)
        sorted_positions = np.append(sorted_positions, self.EOS_token)
        sorted_energies = np.insert(sorted_energies, 0, self.SOS_token)
        sorted_energies = np.append(sorted_energies, self.energy_EOS_token)

        return sorted_positions, sorted_energies, initial_energy
