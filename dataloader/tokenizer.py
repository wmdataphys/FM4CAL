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

    def decode_hits(self,tokens, energies, grid_size=30, SOS_token=0, EOS_token=27001,PAD_token=27002,ground_truth=False):
        tokens = np.asarray(tokens)
        energies = np.asarray(energies)
        
        # Remove SOS, EOS or PAD
        pixel_mask_tokens = np.array([PAD_token, SOS_token, EOS_token])
        pixel_mask = ~np.isin(tokens, pixel_mask_tokens)
        mask = pixel_mask # must be valid in both pixel and energy - this is taken care of, see generate() function of GPT.py
        
        tokens = tokens[mask] - 1 # tokens are offset by 1
        energies = energies[mask]
        
        # Convert flat token -> (z, y, x)
        z = tokens // (grid_size * grid_size)
        rem = tokens % (grid_size * grid_size)
        y = rem // grid_size
        x = rem % grid_size

        assert energies.any() != -1

        # Safety check
        if np.any(tokens < 0) or np.any(tokens >= grid_size**3):
            raise ValueError("Decoded token out of [0, grid_size^3). Check token definitions.")

        return z,x,y,energies

    def decode(self, tokens, energies, material=None,shift=10,filter=0.1):
        if material == "G4_Ta_e-":
            topk = 1
        elif material == "G4_Pb_e-":
            topk = 3
        else:           
            topk = None

        z,x,y,E = self.decode_hits(tokens, energies)
        E = self.de_tokenize(E)
        mask = E > filter
        z, x, y, E = z[mask], x[mask], y[mask], E[mask]
        
        if topk is not None and E.shape[0] > 5:
            if E.shape[0] > 5: 
                top_k = np.argpartition(E, -topk)[-topk:]  # Get indices of top k energies
                x[top_k] += shift # Shift top k hits back 
                x = np.clip(x, 0, 29) # Ensure we don't go out of bounds

        return z,x,y,E


        
