dependencies = ["torch", "torchaudio", "numpy", "sklearn", "numba"]

URLS = {
    "hubert": {
        "english": "https://github.com/bshall/dusted/releases/download/v0.1/hubert-english-e94164.pt",
        "chinese": "https://github.com/bshall/dusted/releases/download/v0.1/hubert-chinese-5635a7.pt",
        "french": "https://github.com/bshall/dusted/releases/download/v0.1/hubert-french-e195fb.pt",
    },
    "kmeans": {
        "english": "https://github.com/bshall/dusted/releases/download/v0.1/kmeans-english-50f36a.pt",
        "chinese": "https://github.com/bshall/dusted/releases/download/v0.1/kmeans-chinese-9381ef.pt",
        "french": "https://github.com/bshall/dusted/releases/download/v0.1/kmeans-french-2a0b9a.pt",
    },
}

from typing import Tuple, Callable

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
import numpy as np

from sklearn.cluster import KMeans
from dusted import encode, segment, match_rescore


def hubert(language: str = "english", pretrained: bool = True, progress: bool = True):
    r"""HuBERT content encoders from `"Spoken-Term Discovery using Discrete Speech Units"`.
    The english checkpoint is from `"HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"` originally available at https://github.com/facebookresearch/fairseq.
    The chinese checkpoint is originally from https://github.com/TencentGameMate/chinese_speech_pretrain.
    The french checkpoint is the multi-lingual model (trained on French, English, and Spanish) from `"Textless Speech-to-Speech Translation on Real Data"` originally available at https://github.com/facebookresearch/fairseq.

    Args:
        language (str): the pre-training language of the HuBERT checkpoint (choose from english, chinese, or french).
        pretrained (bool): load pretrained weights into the model.
        progress (bool): show progress bar when downloading model.

    Returns:
        Hubert: the HuBERT model (see https://github.com/bshall/hubert)
        Callable: a helper function for extracting features from intermediate layers.
    """
    model = torch.hub.load("bshall/hubert:main", "hubert", trust_repo=True)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS["hubert"][language], progress=progress
        )
        consume_prefix_in_state_dict_if_present(checkpoint, "module.")
        model.load_state_dict(checkpoint)
        model.eval()
    return model, encode


def kmeans(
    language: str = "english", pretrained: bool = True, progress: bool = True
) -> KMeans:
    r"""k-means checkpoint with 100 clusters for the HuBERT content encoders from `"Spoken-Term Discovery using Discrete Speech Units"`.

    Args:
        language (str): the raining language of the k-means checkpoint (choose from english, chinese, or french).
        pretrained (bool): load pretrained weights into the model.
        progress (bool): show progress bar when downloading model.

    Returns:
        KMeans: the k-means model.
        Callable: the segmentation function to group HuBERT features into phone-like units.
    """
    model = KMeans(100)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS["kmeans"][language], progress=progress
        )
        model.__dict__["n_features_in_"] = checkpoint["n_features_in_"]
        model.__dict__["_n_threads"] = checkpoint["_n_threads"]
        model.__dict__["cluster_centers_"] = checkpoint["cluster_centers_"].numpy()
    return model, segment


def dusted() -> Tuple[np.ndarray, Callable]:
    r"""The similariy matrix and matching function from `"Spoken-Term Discovery using Discrete Speech Units"`

    Returns:
        NDArray: similarity matrix where sim[i, j] returns the score for substituting unit i with unit j.
        Callable: the dynamic programming algorithm to find matching unit sub-sequences.
    """
    sim = np.full((100, 100), -1, dtype=np.float32)
    return sim, match_rescore
