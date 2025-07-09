import numpy as np


class EnergyTokenizer():
    def __init__(self, e_max, e_min, resolution=0.01):
        super().__init__()
        self.e_max = e_max
        self.e_min = e_min
        self.energy_res = resolution
        self.e_bins = np.arange(self.e_min,
                                self.e_max + self.energy_res,
                                self.energy_res)

    def tokenize(self, energies):
        """
        Supports tokenization of both dense and sparse array representations

        - Dense Input: np.ndarray of shape (30, 30, 30)
        - Sparse input: tuple of (indices, values)

        Returns:
        - Tokens of shape (27000,) with digitized energy values
        """
        # Sparse tuple input
        if isinstance(energies, tuple) and len(energies) == 2:
            shape = (30, 30, 30)
            indices, values = energies
            flat_indices = np.ravel_multi_index(indices.T, shape)
            flat_array = np.zeros(np.prod(shape), dtype=values.dtype)
            flat_array[flat_indices] = values
            return np.digitize(flat_array, self.e_bins) + 1  # Offset so SOS = 0

        # Dense array, (30, 30, 30)
        if isinstance(energies, np.ndarray):
            return np.digitize(energies.flatten(), self.e_bins) + 1  # Offset so SOS = 0

        else:
            raise ValueError("Unsupported input format for tokenization")

    def de_tokenize(self, tokens):
        z = self.e_min + (tokens - 1 + 0.5) * self.energy_res
        # sample with time resolution
        # z = z + np.random.normal(loc=0,
        #                           scale=self.energy_res * 0.5,
        #                           size=tokens.shape)
        z = z + np.random.uniform(-0.5 * self.energy_res,
                                  0.5 * self.energy_res,
                                  size=tokens.shape)
        return np.clip(z, self.e_min, self.e_max)
