import argparse
from pathlib import Path

import random
import numpy as np
import torch
from sklearn.cluster import KMeans


def cluster(args):
    paths = list(args.in_dir.rglob("*.npy"))
    random.shuffle(paths)

    xs = []
    duration = 0
    hours = round(args.hours * 60 * 60 / 0.02)

    print(f"Loading features ({args.hours} hours)...")
    for path in paths:
        x = np.load(path)
        duration += x.shape[0]
        xs.append(x)

        if duration > hours:
            break

    xs = np.concatenate(xs, axis=0)

    print(f"Clustering with {args.clusters} centriods...")
    kmeans = KMeans(args.clusters)
    kmeans.fit(xs)

    args.out_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(
        {
            "n_features_in_": kmeans.n_features_in_,
            "_n_threads": kmeans._n_threads,
            "cluster_centers_": torch.from_numpy(kmeans.cluster_centers_),
        },
        args.out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster HuBERT features.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        type=Path,
        help="path to the speech features.",
    )
    parser.add_argument(
        "out_path",
        metavar="out-path",
        type=Path,
        help="path to the output checkpoint",
    )
    parser.add_argument(
        "--clusters",
        default=100,
        type=int,
        help="number of clusters.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        help="number of hours of speech to use (defaults to 5).",
        default=5,
    )
    args = parser.parse_args()
    cluster(args)
