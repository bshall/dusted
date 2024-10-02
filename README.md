# DUSTED: Spoken-Term Discovery using Discrete Speech Units

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2408.14390)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bshall/dusted/blob/main/demo.ipynb)

Official repository for [Spoken-Term Discovery using Discrete Speech Units](https://arxiv.org/abs/2408.14390).

<div align="center">
    <img width="100%" alt="DUSTED"
      src="https://raw.githubusercontent.com/bshall/dusted/main/dusted.svg">
</div>
<div>
  <sup>
    <strong>Fig 1:</strong> Overview of DUSTED: Discrete Unit Spoken-Term Discovery. First, the  <strong>content encoder</strong> maps a pair of input utterances to sequences of discrete units. Then, the <strong>pattern matcher</strong> searches for similar unit sub-sequences to find shared words or phrases in the inputs.
  </sup>
</div>

**Abstract**: Discovering a lexicon from unlabeled audio is a longstanding challenge for zero-resource speech processing. 
One approach is to search for frequently occurring patterns in speech. 
We revisit this idea with DUSTED: **D**iscrete **U**nit **S**poken-**TE**rm **D**iscovery.
Leveraging self-supervised models, we encode input audio into sequences of discrete units. 
Next, we find repeated patterns by searching for similar unit sub-sequences, inspired by alignment algorithms from bioinformatics. 
Since discretization discards speaker information, DUSTED finds better matches across speakers, improving the coverage and consistency of the discovered patterns. 
We demonstrate these improvements on the ZeroSpeech Challenge, achieving state-of-the-art results on the spoken-term discovery track. 
Finally, we analyze the duration distribution of the patterns, showing that our method finds longer word- or phrase-like terms.

## Example Usage

### Programmatic Usage

```python
import torch, torchaudio

# Load the Hubert content encoder (see https://github.com/bshall/hubert/)
hubert, encode = torch.hub.load("bshall/dusted:main", "hubert", language="english", trust_repo=True)
hubert.cuda()

# Load the k-means checkpoint
kmeans, segment = torch.hub.load("bshall/dusted:main", "kmeans", language="english", trust_repo=True)

# Load the similarity function and pattern matcher
sim, match = torch.hub.load("bshall/dusted:main", "dusted", trust_repo=True)

# Load the pair of audio clips
xwav, sr = torchaudio.load("path/to/xwav")
ywav, sr = torchaudio.load("path/to/ywav")

xwav = xwav.unsqueeze(0).cuda()
ywav = ywav.unsqueeze(0).cuda()

# Encode the audio
x = encode(hubert, xwav).squeeze().cpu().numpy()
y = encode(hubert, ywav).squeeze().cpu().numpy()

# Segment the features into phone-like units
xcodes, xboundaries = segment(x, kmeans.cluster_centers_, gamma=0.2)
ycodes, yboundaries = segment(y, kmeans.cluster_centers_, gamma=0.2)

# Search for matching unit sub-sequences
for path, a, b, similarity in match(xcodes, ycodes, sim, gap=1, threshold=6):
    # Find start and end times of the matching sub-sequences
    a0 = round(xboundaries[a0 - 1] * 0.02, 2)
    an = round(xboundaries[an] * 0.02, 2)
    b0 = round(yboundaries[b0 - 1] * 0.02, 2)
    bn = round(yboundaries[bn] * 0.02, 2)
    # Write to file (or other processing)
```

### Script-Based Usage

#### Step 1: Extract HuBERT Features

We recommend applying VAD to the audio dataset before the content encoding and pattern matching steps.

```
usage: encode.py [-h] [--layer LAYER] [--extension EXTENSION]
                 in-dir out-dir {english,chinese,french}

Encode an audio dataset using HuBERT.

positional arguments:
  in-dir                path to the dataset directory.
  out-dir               path to the output directory.
  {english,chinese,french}
                        pre-training language of the HuBERT content encoder.

options:
  -h, --help            show this help message and exit
  --layer LAYER         HuBERT layer to extract features from (defaults to 7).
  --extension EXTENSION
                        extension of the audio files (defaults to .wav).

```

#### Step 2: Segment the Features into Longer Units

```
usage: segment.py [-h] [--gamma GAMMA] [--processes PROCESSES]
                  in-dir out-dir {english,chinese,french}

Segment an audio dataset into phone-like units.

positional arguments:
  in-dir                path to the speech features.
  out-dir               path to the output directory.
  {english,chinese,french}
                        pre-training language of the HuBERT content encoder.

options:
  -h, --help            show this help message and exit
  --gamma GAMMA         regularization weight for segmentation (defaults to
                        0.2).
  --processes PROCESSES
                        number of processes (defaults to 10).
```

#### Step 3: Find Matching Unit Sub-sequences

```
usage: match.py [-h] [--gap GAP] [--threshold THRESHOLD]
                [--min_duration MIN_DURATION] [--processes PROCESSES]
                [--chunksize CHUNKSIZE]
                segments-dir out-path

Find matching audio fragments in a dataset.

positional arguments:
  segments-dir          path to the directory of segmented audio.
  out-path              path to the output csv.

options:
  -h, --help            show this help message and exit
  --gap GAP             gap cost.
  --threshold THRESHOLD
                        minimum score required for a match (defaults to 6).
  --min_duration MIN_DURATION
                        minimum duration required for a match (defaults to 0.2
                        seconds)
  --processes PROCESSES
                        number of processes (defaults to 10).
  --chunksize CHUNKSIZE
                        multiprocessing chunksize (defaults to 200).
```

## Applying DUSTED to the ZeroSpeech Challenge

1. Install the Zerospeech benchmark toolkit:

```
pip install zerospeech-benchmarks[all]
```

2. Download the Zerospeech 2017 datasets:

```
zrc datasets:pull zrc2017-test-dataset
zrc datasets:pull zrc2017-train-dataset
```

3. Split the dataset by the provided VAD marks using the `preprocess` script:
```
usage: preprocess.py [-h] in-dir out-dir vad-path

Preprocess the ZeroSpeech 2017 datasets by splitting the audio according to
the vad marks.

positional arguments:
  in-dir      path to the dataset directory.
  out-dir     path to the output directory.
  vad-path    path to the VAD csv.

options:
  -h, --help  show this help message and exit

```

4. Extract HuBERT features using the `encode` script (see above).

5. Segment the HuBERT features into longer units using the `segment` script.

6. Search for matching unit subsequences using the `match` script.

7. Download and extract the `submission` folder [here](https://github.com/bshall/dusted/releases/download/v0.1/submission.zip). Since the toolkit requires all languages for validation this folder contains dummy files to allow you to evaluate just a single language instead.

8. Format the `pairs.csv` file for evaluation using the `format` script (note this requires python 3.12):
```
usage: format.py [-h] [--threshold THRESHOLD] pairs-path submission-path

Format the csv of pairs for evaluation.

positional arguments:
  pairs-path            path to the csv of pairs.
  submission-path       path to the txt file for submission.

options:
  -h, --help            show this help message and exit
  --threshold THRESHOLD
                        the threshold for including a pair (defaults to 6)
```

9. Run the Zerospeech evaluation:
```
zrc benchmarks:run tde17 path/to/submission/
```
The results will be output to `path/to/submission/scores/scores.json`.

The grouping metric can take a long time to run and can use up a lot or memory so you can skip it by hitting `ctrl+c` when you see `computing grouping for english...`

10. We also include the `cluster.py` script to train a k-means model on your own data:
```
usage: cluster.py [-h] [--clusters CLUSTERS] [--hours HOURS] in-dir out-path

Cluster HuBERT features.

positional arguments:
  in-dir               path to the speech features.
  out-path             path to the output checkpoint

options:
  -h, --help           show this help message and exit
  --clusters CLUSTERS  number of clusters.
  --hours HOURS        number of hours of speech to use (defaults to 5).
```