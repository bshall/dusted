# DUSTED: Spoken-Term Discovery using Discrete Speech Units

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2408.14390)

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
sim, match = torch.hub.load("bshall/dusted:main", "kmeans", trust_repo=True)

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
for path, a, b, similarity in match_rescore(xcodes, ycodes, sim, gap=1, threshold=6):
    # Find start and end times of the matching sub-sequences
    a0 = round(xboundaries[a0 - 1] * 0.02, 2)
    an = round(xboundaries[an] * 0.02, 2)
    b0 = round(yboundaries[b0 - 1] * 0.02, 2)
    bn = round(yboundaries[bn] * 0.02, 2)
    # Write to file (or other processing)
```