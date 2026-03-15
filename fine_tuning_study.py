import os
# Set these BEFORE importing torch
os.environ["TORCH_LOGS"] = "-cudagraphs" # Explicitly subtract cudagraphs from logs
os.environ["GLOG_minloglevel"] = "3"      # Suppress C++ Google Logs (3 = Fatal)

import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import h5py
from pathlib import Path
import pkbar
from datetime import datetime
import argparse

from plotting import make_fine_tune_hists,plot_fine_tune_comparison

from dataloader.tokenizer import EnergyTokenizer


def main(config,args):
    energy_res = config['stats']['token_energy_res']
    e_max = config['stats']['token_energy_max']
    e_min = config['stats']['token_energy_min']
    material_list = config['material_list']

    dataset_sizes = args.dataset_sizes
    energy_digitizer = EnergyTokenizer(e_max=e_max, e_min=e_min, resolution=energy_res)

    datasets = {}
    for dataset_size in dataset_sizes:
        full_path = config['Inference']['full_fine_tune_path'] if "Full" in dataset_size else None

        result = make_fine_tune_hists(args.base_dir, energy_digitizer,material_list=material_list, material_to_plot=args.material_to_plot, dataset_size=dataset_size,full_path=full_path)
        datasets[dataset_size] = result

    plot_fine_tune_comparison(datasets, material_list, args.material_to_plot, dataset_sizes, output_dir="FineTuningStudies")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate showers using trained GPT model.")
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file.')
    parser.add_argument('--dataset_sizes', type=str, nargs='+', default=None, help='List of dataset sizes to use.')
    parser.add_argument('--material_to_plot', type=str, default=None, help='Material to plot histograms for.')
    parser.add_argument('--output_file', type=str, default=None, help='Output file name for generated showers. Overrides config if set.')
    parser.add_argument('--base_dir', type=str, default=None, help='Base directory for fine-tuning studies. Overrides config if set.')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config, args)