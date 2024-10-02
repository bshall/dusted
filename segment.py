import argparse
from pathlib import Path

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

import torch
import numpy as np


def process_file(paths, codebook, segment, gamma):
    in_path, out_path = paths
    sequence = np.load(in_path)
    codes, boundaries = segment(sequence, codebook, gamma)
    np.savez(out_path.with_suffix(".npz"), codes=codes, boundaries=boundaries)
    return sequence.shape[0], np.mean(np.diff(boundaries))


def segment_dataset(args):
    kmeans, segment = torch.hub.load(
        "bshall/dusted:main", "kmeans", language=args.language, trust_repo=True
    )

    in_paths = list(args.in_dir.rglob("*.npy"))
    out_paths = [args.out_dir / path.relative_to(args.in_dir) for path in in_paths]

    segment_file = partial(
        process_file,
        codebook=kmeans.cluster_centers_,
        segment=segment,
        gamma=args.gamma,
    )

    for path in tqdm(out_paths):
        path.parent.mkdir(exist_ok=True, parents=True)

    print("Segmenting dataset...")
    with Pool(processes=args.processes) as pool:
        results = [
            result
            for result in tqdm(
                pool.imap(
                    segment_file,
                    zip(in_paths, out_paths),
                ),
                total=len(in_paths),
            )
        ]

    frames, boundary_length = zip(*results)
    print(f"Segmented {sum(frames) * 0.02 / 60 / 60:.2f} hours of audio")
    print(f"Average segment length: {np.mean(boundary_length) * 0.02:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment an audio dataset into phone-like units."
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        type=Path,
        help="path to the speech features.",
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        type=Path,
        help="path to the output directory.",
    )
    parser.add_argument(
        "language",
        choices=["english", "chinese", "french"],
        help="pre-training language of the HuBERT content encoder.",
    )
    parser.add_argument(
        "--gamma",
        default=0.2,
        type=float,
        help="regularization weight for segmentation (defaults to 0.2).",
    )
    parser.add_argument(
        "--processes",
        type=int,
        help="number of processes (defaults to 10).",
        default=10,
    )
    args = parser.parse_args()
    segment_dataset(args)
