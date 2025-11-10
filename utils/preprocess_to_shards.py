#!/usr/bin/env python3
import os, glob
from pathlib import Path
import argparse
import numpy as np
import h5py
from tqdm import tqdm
import webdataset as wds

# Import your tokenizer (adjust path if needed)
from dataloader.tokenizer import EnergyTokenizer


def make_tokens(
    group,
    energy_digitizer,
    max_seq_length: int,
    sos_pos: int = 0,
    eos_pos: int = 27001,
    eos_en: int = 24939,
    sort_by: str = "token",  # "token" or "value"
):
    """Tokenize, sort/trim, then add SOS/EOS and return arrays."""
    indices = group["indices"][()]  # (N, 3)
    values = group["values"][()]    # (N,)
    initE = float(group.attrs["initial_energy"])

    energy_tokens = energy_digitizer.tokenize((indices, values))  # (N,)

    # Choose sort key: token id or original energy values
    if sort_by == "value":
        sort_source = values
    else:
        sort_source = energy_tokens

    # Keep top-K by chosen magnitude, descending
    if max_seq_length < sort_source.size:
        topk = np.argpartition(sort_source, -max_seq_length)[-max_seq_length:]
        pos_sorted = topk[np.argsort(-sort_source[topk])]
    else:
        pos_sorted = np.argsort(-sort_source)

    en_sorted = energy_tokens[pos_sorted]

    # Trim at first energy token == 1 (sentinel); robust to "not found"
    mask = (en_sorted == 1)
    cut = int(mask.argmax()) + 1 if mask.any() else len(en_sorted)
    pos_sorted = pos_sorted[:cut]
    en_sorted = en_sorted[:cut]

    # Add SOS/EOS and cast to compact types
    pos_tokens = np.concatenate([[sos_pos], pos_sorted, [eos_pos]]).astype(np.uint16)
    en_tokens = np.concatenate([[sos_pos], en_sorted, [eos_en]]).astype(np.uint16)
    initE_arr = np.array([initE], dtype=np.float32)

    return pos_tokens, en_tokens, initE_arr


def h5_iter_samples(h5_paths, energy_digitizer, max_seq_length, sos_pos, eos_pos, eos_en, sort_by):
    for p in h5_paths:
        with h5py.File(p, "r") as f:
            for key in f.keys():
                g = f[key]
                if "indices" in g and "values" in g and "initial_energy" in g.attrs:
                    pos, en, initE = make_tokens(
                        g,
                        energy_digitizer,
                        max_seq_length=max_seq_length,
                        sos_pos=sos_pos,
                        eos_pos=eos_pos,
                        eos_en=eos_en,
                        sort_by=sort_by,
                    )
                    yield {
                        "__key__": f"{Path(p).stem}-{key}",
                        "pos.npy": pos,
                        "en.npy": en,
                        "initE.npy": initE,
                    }


def write_shards(h5_paths, out_dir, energy_digitizer, max_seq_length=1700, maxcount=50000,
                 sos_pos=0, eos_pos=27001, eos_en=24939, sort_by="token"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "shard-%06d.tar")

    n_samples = 0
    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        for sample in tqdm(
            h5_iter_samples(
                h5_paths,
                energy_digitizer,
                max_seq_length=max_seq_length,
                sos_pos=sos_pos,
                eos_pos=eos_pos,
                eos_en=eos_en,
                sort_by=sort_by,
            ),
            desc="Writing shards",
        ):
            sink.write(sample)
            n_samples += 1

    print(f"Wrote {n_samples} samples into shards at: {out_dir}")


def expand_inputs(inputs):
    """Expand a list of globs/paths/dirs into a sorted list of .hdf5 files."""
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(glob.glob(str(p / "*.hdf5"))))
        else:
            # treat as a glob or a file
            matches = glob.glob(item)
            if matches:
                paths.extend(sorted(matches))
            elif p.suffix.lower() == ".hdf5" and p.exists():
                paths.append(str(p))
    # de-dupe while preserving order
    seen = set()
    uniq = []
    for x in paths:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess HDF5 → WebDataset shards")
    ap.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="Input HDF5 globs/paths/dirs (space-separated). Example: /data/train/*.hdf5 /data/more/*.hdf5",
    )
    ap.add_argument(
        "--outdir",
        "-o",
        required=True,
        help="Output directory for shards (will be created).",
    )
    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=27000,
        help="Max number of hits/tokens kept per sample (before adding SOS/EOS).",
    )
    ap.add_argument(
        "--maxcount",
        type=int,
        default=50000,
        help="Max samples per shard tar.",
    )
    ap.add_argument(
        "--sort-by",
        choices=["token", "value"],
        default="token",
        help="Sort by token id (default) or by raw energy value.",
    )
    ap.add_argument("--sos-pos", type=int, default=0, help="SOS token id for position stream.")
    ap.add_argument("--eos-pos", type=int, default=27001, help="EOS token id for position stream.")
    ap.add_argument("--eos-en", type=int, default=24939, help="EOS token id for energy stream.")

    # EnergyTokenizer params
    ap.add_argument("--e-min", type=float, default=1.7589282e-18, help="Minimum energy for digitizer.")
    ap.add_argument("--e-max", type=float, default=87.27755, help="Maximum energy for digitizer.")
    ap.add_argument("--energy-res", type=float, default=0.0035, help="Energy resolution for digitizer.")

    return ap.parse_args()


def main():
    args = parse_args()

    h5_paths = expand_inputs(args.input)
    if not h5_paths:
        raise FileNotFoundError(f"No .hdf5 files found for inputs: {args.input}")

    print(f"[*] Found {len(h5_paths)} HDF5 files.")
    # Build your tokenizer
    energy_digitizer = EnergyTokenizer(e_max=args.e_max, e_min=args.e_min, resolution=args.energy_res)

    write_shards(
        h5_paths=h5_paths,
        out_dir=args.outdir,
        energy_digitizer=energy_digitizer,
        max_seq_length=args.max_seq_length,
        maxcount=args.maxcount,
        sos_pos=args.sos_pos,
        eos_pos=args.eos_pos,
        eos_en=args.eos_en,
        sort_by=args.sort_by,
    )


if __name__ == "__main__":
    main()
