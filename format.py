import argparse
from pathlib import Path

import csv
import itertools
from tqdm import tqdm


def format_submission(args):
    args.submission_path.parent.mkdir(exist_ok=True, parents=True)
    with open(args.submission_path, "w") as file:
        with open(args.pairs_path) as pairs:
            reader = csv.reader(pairs)
            data = itertools.batched(itertools.islice(reader, 1, None), 2)
            for i, pair in tqdm(enumerate(data)):
                score = float(pair[0][-1])
                if score < args.threshold:
                    continue

                file.write(f"Class {i + 1}\n")
                for name, t0, tn, _, score in pair:
                    name, offset = name.split("-")
                    offset = round(int(offset) / 16000, 2)
                    start = round(offset + float(t0), 2)
                    end = round(offset + float(tn), 2)
                    file.write(f"{name} {start} {end}\n")
                file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format the csv of pairs for evaluation."
    )
    parser.add_argument(
        "pairs_path",
        metavar="pairs-path",
        type=Path,
        help="path to the csv of pairs.",
    )
    parser.add_argument(
        "submission_path",
        metavar="submission-path",
        type=Path,
        help="path to the txt file for submission.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="the threshold for including a pair (defaults to 6)",
        default=6,
    )
    args = parser.parse_args()
    format_submission(args)
