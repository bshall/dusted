import torch
import torch.nn.functional as F


@torch.inference_mode()
def encode(hubert, wav, layer=7):
    r"""Encode an audio waveform into Hubert features.

    Args:
        hubert (Hubert): the Hubert encoder.
        wav (Tensor): an audio waveform of shape (B, 1, T) where B is the batch size and T is the number of samples.
        layer (int): the layer to extract features from (defaults to 7).

    Returns:
        Tensor: Hubert features of shape (B, N, D) where N is the number of frames, and D is the feature dimensions.
    """
    wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
    x, _ = hubert.encode(wav, layer=layer)
    return x
