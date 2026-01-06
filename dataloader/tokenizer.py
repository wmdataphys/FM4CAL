import numpy as np

class EnergyTokenizer():
    def __init__(self, e_max=35.0, e_min=1e-15, resolution=0.0014):
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
        - Energy tokens [1, 25000] from np.digitize()
        """
        # Sparse tuple input: only tokenize the energy values
        if isinstance(energies, tuple) and len(energies) == 2:
            indices, values = energies
            return np.digitize(values, self.e_bins)  # [1, 25000]

        # Dense array, (30, 30, 30)
        if isinstance(energies, np.ndarray):
            return np.digitize(energies.flatten(), self.e_bins)  # [1, 25000]

        else:
            raise ValueError("Unsupported input format for tokenization")


    def de_tokenize(self, tokens):
        z = self.e_min + (tokens + 0.5) * self.energy_res
        z = z + np.random.uniform(-0.5 * self.energy_res,
                                  0.5 * self.energy_res,
                                  size=tokens.shape)
        return np.clip(z, self.e_min, self.e_max)
