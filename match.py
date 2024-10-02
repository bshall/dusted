import argparse
from pathlib import Path

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, Manager, Process

import torch
import numpy as np
import math
import itertools
import csv


def match(args):
    paths = list(args.segments_dir.rglob("*.npz"))
    combinations = itertools.combinations(paths, 2)

    sim, match = torch.hub.load(
        "bshall/dusted:main", "dusted", trust_repo=True, force_reload=True
    )

    with Pool(processes=args.processes) as pool, Manager() as manager:
        queue = manager.Queue()

        writer = Process(target=write_results_to_file, args=(args.out_path, queue))
        writer.start()

        match_pair = partial(
            process_pair,
            match=match,
            sim=sim,
            gap=args.gap,
            threshold=args.threshold,
            min_duration=args.min_duration,
        )

        for result in tqdm(
            pool.imap(match_pair, combinations, chunksize=args.chunksize),
            total=math.comb(len(paths), 2),
        ):
            if result:
                queue.put(result)

        queue.put("DONE")
        writer.join()


def process_pair(pair, match, sim, gap=1, threshold=6, min_duration=0.2):
    x_path, y_path = pair
    x_segments = np.load(x_path)
    y_segments = np.load(y_path)

    x_codes, x_boundaries = x_segments["codes"], x_segments["boundaries"]
    y_codes, y_boundaries = y_segments["codes"], y_segments["boundaries"]

    matches = []
    for path, a, b, score in match(x_codes, y_codes, sim, gap, threshold):
        a0, b0 = path[0]
        an, bn = path[-1]

        a0 = round(x_boundaries[a0 - 1] * 0.02, 2)
        an = round(x_boundaries[an] * 0.02, 2)
        b0 = round(y_boundaries[b0 - 1] * 0.02, 2)
        bn = round(y_boundaries[bn] * 0.02, 2)

        if an - a0 <= min_duration or bn - b0 <= min_duration:
            continue

        atokens = tuple(str(token) for token in a if token != -1)
        btokens = tuple(str(token) for token in b if token != -1)
        atokens = " ".join(atokens)
        btokens = " ".join(btokens)

        matches.append((x_path.stem, a0, an, atokens, score))
        matches.append((y_path.stem, b0, bn, btokens, score))

    return matches


def write_results_to_file(out_path, queue):
    out_path.parent.mkdir(exist_ok=True, parents=True)

    with open(out_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(["file", "t0", "tn", "tokens", "score"])

        while True:
            result = queue.get()
            if result == "DONE":
                break

            writer.writerows(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find matching audio fragments in a dataset."
    )
    parser.add_argument(
        "segments_dir",
        metavar="segments-dir",
        type=Path,
        help="path to the directory of segmented audio.",
    )
    parser.add_argument(
        "out_path",
        metavar="out-path",
        type=Path,
        help="path to the output csv.",
    )
    parser.add_argument(
        "--gap",
        type=float,
        help="gap cost.",
        default=1,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="minimum score required for a match (defaults to 6).",
        default=6,
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        help="minimum duration required for a match (defaults to 0.2 seconds)",
        default=0.2,
    )
    parser.add_argument(
        "--processes",
        type=int,
        help="number of processes (defaults to 10).",
        default=10,
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        help="multiprocessing chunksize (defaults to 200).",
        default=200,
    )
    args = parser.parse_args()
    match(args)
