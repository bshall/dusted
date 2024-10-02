import argparse
from pathlib import Path

from tqdm import tqdm
import itertools
import csv
import soundfile


def preprocess(args):
    with open(args.vad_path) as file:
        reader = csv.reader(file)
        vad = (
            [name, float(start), float(end)]
            for name, start, end in itertools.islice(reader, 1, None)
        )
        vad = ([name, start, end] for name, start, end in vad if end - start >= 0.2)
        vad = [[name, int(start * 16000), int(end * 16000)] for name, start, end in vad]

    for name, start, end in tqdm(vad):
        if name == "s0466":
            continue
        in_path = args.in_dir / name
        wav, sr = soundfile.read(in_path.with_suffix(".wav"), start=start, stop=end)

        out_path = args.out_dir / f"{name}-{start}.wav"
        out_path.parent.mkdir(exist_ok=True, parents=True)
        soundfile.write(out_path, wav, sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess the ZeroSpeech 2017 datasets by splitting the audio according to the vad marks."
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        type=Path,
        help="path to the dataset directory.",
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        type=Path,
        help="path to the output directory.",
    )
    parser.add_argument(
        "vad_path",
        metavar="vad-path",
        type=Path,
        help="path to the VAD csv.",
    )
    args = parser.parse_args()
    preprocess(args)
