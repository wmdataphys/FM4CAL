import h5py
import os
import csv

val_data = "/sciclone/data10/cjgranger/ECAL/ECAL_Cole/ECALSim/ILDConfig/StandardConfig/production/simulation/processed/val"
output = "/sciclone/home/cjgranger/FM4CAL/val_init_e.csv"

init_energies = []
for file_path in os.listdir(val_data):
    with h5py.File(os.path.join(val_data, file_path), "r") as f:
        keys = f.keys()

        for key in keys:
            group = f[key]
            initial_energy = group.attrs['initial_energy']
            init_energies.append(initial_energy)

with open(output, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(init_energies)
