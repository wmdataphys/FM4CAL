import numpy as np
import h5py


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
