import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
from torchaudio.functional import resample


def encode_dataset(args):
    print(f"Loading hubert checkpoint")
    hubert, encode = torch.hub.load(
        "bshall/dusted:main", "hubert", language=args.language, trust_repo=True
    )
    hubert.cuda()

    print(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        wav, sr = torchaudio.load(in_path)
        wav = resample(wav, sr, 16000)
        wav = wav.unsqueeze(0).cuda()

        x = encode(hubert, wav, args.layer)

        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), x.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode an audio dataset using HuBERT."
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "language",
        choices=["english", "chinese", "french"],
        help="pre-training language of the HuBERT content encoder.",
    )
    parser.add_argument(
        "--layer",
        help="HuBERT layer to extract features from (defaults to 7).",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .wav).",
        default=".wav",
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)
