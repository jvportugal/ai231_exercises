"""Long centered context with speech-aware normalization and gentle pooling.

Two input channels: mel, and a broadcast speech/context support mask. The
mask covers the whole detected utterance including internal pauses, plus
200 ms on each side. It is not a word-level VAD or a semantic OOS filter.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from train_temporal_attention import TemporalAttentionVCM

ME2 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ME2))
import vcm_infer as runtime


from vcm_long_window import features


class LongWindowAttentionVCM(TemporalAttentionVCM):
    def __init__(self, width=142, gru_layers=2, auxiliary_words=96,
                 masked=True, center_strength=0.):
        super().__init__(width=width, gru_layers=gru_layers,
                         auxiliary_words=auxiliary_words, structured_slots=True)
        self.masked = masked
        self.center_strength = float(center_strength)

    def temporal_prior(self, support):
        # Normalize position by the supported span, so a long command gets
        # a broad preference, not the narrow bump used for short commands.
        positions = torch.arange(support.shape[1], device=support.device,
                                 dtype=support.dtype)[None]
        mass = support.sum(1, keepdim=True).clamp(min=1.)
        center = (positions * support).sum(1, keepdim=True) / mass
        radius = (mass / 2).clamp(min=1.)
        prior = -self.center_strength * ((positions - center) / radius).square()
        return prior.masked_fill(support < .5, -1e4)

    def encode(self, packed):
        mel = packed[:, :1]
        x = self.cnn(self.stem(mel))
        # Exactly match the three time-stride-2 CNN blocks. Max pooling
        # keeps every step whose receptive neighborhood overlaps support.
        support = packed[:, 1, 0:1]
        for _ in range(3):
            support = F.max_pool1d(support, 3, stride=2, padding=1)
        support = support[:, 0]
        if not self.masked:
            support = torch.ones_like(support)
        # Defensive fallback for direct model callers supplying an empty mask.
        support = torch.where(support.sum(1, keepdim=True) > 0,
                              support, torch.ones_like(support))
        sequence = x.mean(dim=2).transpose(1, 2)
        sequence = sequence * support[:, :, None]
        sequence, _ = self.gru(sequence)
        attended, _ = self.attention(sequence, sequence, sequence,
            key_padding_mask=support < .5, need_weights=False)
        sequence = self.attention_norm(sequence + attended)
        sequence = self.ffn_norm(sequence + self.ffn(sequence))
        prior = self.temporal_prior(support)
        weights = (self.pool_score(sequence).squeeze(-1) + prior).softmax(1)[:, :, None]
        mean = (weights * sequence).sum(1)
        variance = (weights * (sequence - mean[:, None]).square()).sum(1)
        return self.embedding(torch.cat((mean, variance.clamp(min=1e-5).sqrt()), dim=1))


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = LongWindowAttentionVCM(width=cfg["width"], gru_layers=cfg["gru_layers"],
        auxiliary_words=len(payload["vocabulary"]), masked=cfg["masked"],
        center_strength=cfg["center_strength"]).to(device).eval()
    model.load_state_dict(payload["model"])
    return model, payload


def export_model(model, destination, n_frames):
    # Use the decomposed attention path supported by ONNX, then restore the
    # caller's setting. This changes export tracing, not the trained weights.
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        x = torch.zeros(1, 2, 40, n_frames)
        x[:, 1] = 1
        with torch.no_grad():
            torch.onnx.export(model.cpu().eval(), x, str(destination), input_names=["mel"],
                output_names=["intent", "slot", "local", "words"],
                dynamic_axes={n: {0: "batch"} for n in ("mel", "intent", "slot", "local", "words")},
                opset_version=17, dynamo=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)
