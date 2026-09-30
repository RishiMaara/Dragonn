"""
Whisper's decoder on the Hexagon NPU.

The encoder is one fixed-shape pass, which is why it was the easy half. A
decoder is autoregressive: shapes grow with every token, and a fixed-point
accelerator compiles a fixed graph. Qualcomm's answer, published with their AI
Hub Whisper models, is a decoder whose cache is a fixed 199 slots and whose
history is right-aligned — each step shifts the cache left by one and writes the
new key/value at the end, so every tensor keeps its shape forever:

    k_cache_self  [heads, 1, head_dim, 199]      (keys are stored transposed)
    v_cache_self  [heads, 1, 199, head_dim]
    attention_mask[1, 1, 1, 200]                 -100 hides a slot, 0 reveals it

At step n the mask reveals index (200 - n - 1), so the first token attends only
to itself and the window opens leftward as the cache fills.

This module is the loop, not the model: the model is Qualcomm's, downloaded
pre-compiled for the chipset, and the loop is what drives it. Their decoder
measured 2.60 ms per token on a real Snapdragon X Elite with all 509 layers on
the NPU (models/reports/aihub_qualcomm_whisper.json).

Running it end to end needs a physical Snapdragon device: the pre-compiled model
is an EPContext graph that only QNN EP can load, and no cloud service rents one
interactively. What is verified here without that device is the contract —
tests/test_npu_decoder.py builds every input this loop produces and checks it
against the shapes, dtypes and names the published model declares.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("dragonn.npu-decoder")

# Qualcomm's published constants for the Whisper decoder bundles.
MASK_NEG = -100.0          # not float32's min: see converter.quantize.clamp_extreme_constants
MAX_DECODE_LEN = 200       # the mask width; the cache holds one fewer
CACHE_LEN = MAX_DECODE_LEN - 1


@dataclass(frozen=True)
class DecoderShape:
    """Whatever Whisper size the bundle holds, read off its own inputs."""
    layers: int
    heads: int
    head_dim: int
    cache_len: int = CACHE_LEN
    max_decode_len: int = MAX_DECODE_LEN

    @classmethod
    def from_model(cls, model_path: str | Path) -> "DecoderShape":
        import onnx

        graph = onnx.load(str(model_path), load_external_data=False).graph
        shapes = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in graph.input}
        self_keys = [n for n in shapes if n.startswith("k_cache_self_")]
        heads, _, head_dim, cache_len = shapes[self_keys[0]]
        return cls(layers=len(self_keys), heads=heads, head_dim=head_dim, cache_len=cache_len,
                   max_decode_len=shapes["attention_mask"][-1])


def initial_state(shape: DecoderShape) -> dict[str, np.ndarray]:
    """Empty self-attention cache and a mask that hides all of it."""
    state: dict[str, np.ndarray] = {}
    for layer in range(shape.layers):
        state[f"k_cache_self_{layer}_in"] = np.zeros(
            (shape.heads, 1, shape.head_dim, shape.cache_len), np.float16)
        state[f"v_cache_self_{layer}_in"] = np.zeros(
            (shape.heads, 1, shape.cache_len, shape.head_dim), np.float16)
    state["attention_mask"] = np.full((1, 1, 1, shape.max_decode_len), MASK_NEG, np.float16)
    return state


def step_inputs(token: int, position: int, state: dict[str, np.ndarray],
                cross: dict[str, np.ndarray], shape: DecoderShape) -> dict[str, np.ndarray]:
    """
    One decode step's full input dict.

    The mask is revealed in place for this position before the call — index
    (max_decode_len - position - 1), because the cache is right-aligned.
    """
    mask = state["attention_mask"]
    mask[:, :, :, shape.max_decode_len - position - 1] = 0.0
    return {
        "input_ids": np.array([[token]], np.int32),
        "position_ids": np.array([position], np.int32),
        "attention_mask": mask,
        **{k: v for k, v in state.items() if k != "attention_mask"},
        **cross,
    }


def advance(state: dict[str, np.ndarray], outputs: dict[str, np.ndarray],
            shape: DecoderShape) -> dict[str, np.ndarray]:
    """Feed each layer's new cache back in as the next step's input."""
    for layer in range(shape.layers):
        state[f"k_cache_self_{layer}_in"] = outputs[f"k_cache_self_{layer}_out"]
        state[f"v_cache_self_{layer}_in"] = outputs[f"v_cache_self_{layer}_out"]
    return state


class NpuDecoder:
    """
    Drives Qualcomm's pre-compiled Whisper decoder on the Hexagon NPU.

    Requires a Snapdragon device: the model is an EPContext graph, so QNN EP is
    the only execution provider that can load it, and create_session(strict=True)
    refuses rather than quietly falling back to CPU.
    """

    def __init__(self, decoder_path: str | Path, cache_dir: str | Path | None = ".qnn_cache"):
        from runtime.qnn_ep import create_session

        self.path = Path(decoder_path)
        self.shape = DecoderShape.from_model(self.path)
        self.session = create_session(self.path, {"htp_arch": "73"},
                                      cache_dir=cache_dir, strict=True)
        self.output_names = [o.name for o in self.session.get_outputs()]
        logger.info(
            f"Decoder on {self.session.get_providers()[0]}: {self.shape.layers} layers, "
            f"{self.shape.heads} heads, {self.shape.cache_len}-slot cache"
        )

    def logits_name(self) -> str:
        return next(n for n in self.output_names if "cache" not in n)

    def decode(self, cross: dict[str, np.ndarray], prompt: list[int], eot: int,
               max_tokens: int | None = None) -> list[int]:
        """
        Greedy decode. `cross` is the encoder's cross-attention cache, which on a
        Snapdragon comes straight from Qualcomm's encoder bundle — the encoder
        hands the decoder its keys and values rather than hidden states.
        """
        limit = min(max_tokens or self.shape.cache_len, self.shape.cache_len)
        state = initial_state(self.shape)
        tokens = list(prompt)
        logits_name = self.logits_name()

        for position in range(limit):
            token = tokens[position] if position < len(tokens) else tokens[-1]
            feeds = step_inputs(token, position, state, cross, self.shape)
            results = dict(zip(self.output_names, self.session.run(None, feeds)))
            state = advance(state, results, self.shape)

            if position + 1 < len(prompt):
                continue                                  # still feeding the prompt
            nxt = int(np.argmax(results[logits_name][0, -1]))
            if nxt == eot:
                break
            tokens.append(nxt)

        return tokens[len(prompt):]
