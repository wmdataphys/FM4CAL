import torch
import numpy as np
from torch.utils.data import DataLoader


def ECAL_collate_fn(batch):
    positions, energies, initial_energies = zip(*batch)

    max_len = max(len(p) for p in positions)

    padded_positions = []
    padded_energies = []

    for pos, en in zip(positions, energies):
        pad_len = max_len - len(pos)
        padded_positions.append(
            torch.tensor(np.pad(pos, (0, pad_len), constant_values=27002))  # pad_token
        )
        padded_energies.append(
            torch.tensor(np.pad(en, (0, pad_len), constant_values=en[-1] + 1))  # energy_pad_token
        )

    padded_positions = torch.stack(padded_positions)
    padded_energies = torch.stack(padded_energies)
    initial_energies = torch.tensor(initial_energies).float()

    return padded_positions, padded_energies, initial_energies


def CreateECALLoaders(train_dataset, val_dataset, config):
    train_loader = DataLoader(train_dataset,
                            batch_size=config['dataloader']['train']['batch_size'],
                            shuffle=True,
                            collate_fn=ECAL_collate_fn,
                            num_workers=config['dataloader']['train']['num_workers'],
                            pin_memory=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=config['dataloader']['val']['batch_size'],
                            shuffle=False,
                            collate_fn=ECAL_collate_fn,
                            num_workers=config['dataloader']['val']['num_workers'],
                            pin_memory=False)
    return train_loader, val_loader


def CreateLoadersMoE(train_dataset, val_dataset, config):
    train_loader = DataLoader(train_dataset,
                            batch_size=config['dataloader']['train']['batch_size_MoE'],
                            shuffle=True, collate_fn=ECAL_collate_fn, num_workers=config['dataloader']['train']['num_workers'],
                            pin_memory=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=config['dataloader']['val']['batch_size_MoE'],
                            shuffle=False, collate_fn=ECAL_collate_fn, num_workers=config['dataloader']['val']['num_workers'],
                            pin_memory=False)

    return train_loader, val_loader


if __name__ == '__main__':
    from dataset import ECAL_Dataset
    from tokenizer import EnergyTokenizer

    min_e = np.float64(3.637978807091713e-11)
    max_e = np.float64(62.29130172729492)

    energy_digitizer = EnergyTokenizer(max_e, min_e)

    dataset = ECAL_Dataset('C:\\Users\\cjgra\\Documents\\FM4PHYS\\FM4CAL\\sparse_photons.hdf5',
                           energy_digitizer=energy_digitizer)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=ECAL_collate_fn)

    for batch_idx, (pos_batch, en_batch, init_energy_batch) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Position tensor shape: {pos_batch.shape}")
        print(f"  Energy tensor shape:   {en_batch.shape}")
        print(f"  Initial energies:      {init_energy_batch}")
        break  # Just test the first batch
