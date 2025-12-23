import torch
import numpy as np
from torch.utils.data import DataLoader
from functools import partial

# pos, ene, initial_energy, material_index, pos_t, ene_t, initial_energy_t, material_index_t

def ECAL_collate_inference(batch, max_seq_length=1700):
    positions = []
    energies = []
    initial_energies = []
    material_indices = []
    initial_energy_t = []
    energy_copies = []

    for pos, ene, initial_energy, material_index, init_e_t, energy_copy in batch:
        # Already padded
        positions.append(torch.tensor(pos))
        energies.append(torch.tensor(ene))
        initial_energies.append(torch.tensor(initial_energy))
        material_indices.append(torch.tensor(material_index))

        initial_energy_t.append(torch.tensor(init_e_t)) # Unscaled eenergy
        energy_copies.append(torch.tensor(energy_copy)) # Ground truth energy copy


    return torch.stack(positions), torch.stack(energies), torch.tensor(initial_energies), torch.stack(material_indices), \
           torch.tensor(initial_energy_t), torch.stack(energy_copies)

def CreateInferenceLoader(dataset,config):
    loader = DataLoader(dataset,
                            batch_size=config['Inference']['batch_size'],
                            shuffle=False,
                            collate_fn=ECAL_collate_inference,
                            num_workers=config['Inference']['num_workers'],
                            pin_memory=False)
    return loader

def ECAL_collate_fn(batch, max_seq_length=1700):
    positions = []
    energies = []
    initial_energies = []
    material_indices = []

    for pos, ene, initial_energy, material_index in batch:
        # Already padded
        positions.append(torch.tensor(pos))
        energies.append(torch.tensor(ene))
        initial_energies.append(torch.tensor(initial_energy))
        material_indices.append(torch.tensor(material_index))

    return torch.stack(positions), torch.stack(energies), torch.tensor(initial_energies), torch.stack(material_indices)

def CreateLoaderMoE(dataset,sampler,batch_size=256,num_workers=8,pin_memory=True,persistent_workers=False,prefetch_factor=2):
    loader = DataLoader(dataset,sampler=sampler,
                            batch_size=batch_size,
                            collate_fn=ECAL_collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers,
                            prefetch_factor=prefetch_factor)
    return loader


def CreateECALLoaders(train_dataset, val_dataset, config):

    collate_fn = partial(ECAL_collate_fn, max_seq_length=config['model']['max_seq_length'])
    train_loader = DataLoader(train_dataset,
                            batch_size=config['dataloader']['train']['batch_size'],
                            shuffle=True,
                            collate_fn=collate_fn,
                            num_workers=config['dataloader']['train']['num_workers'],
                            pin_memory=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=config['dataloader']['val']['batch_size'],
                            shuffle=False,
                            collate_fn=collate_fn,
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

def CreateDistLoader(dataset,sampler,batch_size=256,num_workers=8,pin_memory=True,persistent_workers=False):
    loader = DataLoader(dataset,sampler=sampler,
                            batch_size=batch_size,
                            collate_fn=ECAL_collate_fn,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers,
                            drop_last=True)
    return loader


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
