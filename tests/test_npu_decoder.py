"""
The NPU decode loop, checked against the contract the shipped model declares.

Running the decoder end to end needs a physical Snapdragon: it is an EPContext
graph only QNN EP can load. What can be checked anywhere is that every tensor
this loop builds matches what the model asks for — which is where a decoder
integration actually goes wrong (a transposed key cache, a mask that hides the
current token, an off-by-one position).

The contract below is Qualcomm's published Whisper-Tiny decoder for Snapdragon
X Elite, whose real shapes are in models/reports/aihub_qualcomm_whisper.json.
"""

import numpy as np
import pytest

from speech.npu_decoder import (
    CACHE_LEN,
    MASK_NEG,
    MAX_DECODE_LEN,
    DecoderShape,
    advance,
    initial_state,
    step_inputs,
)

# whisper-tiny: 4 decoder layers, 6 heads, 64 per head.
TINY = DecoderShape(layers=4, heads=6, head_dim=64)

# name -> (shape, dtype), exactly as the published decoder.onnx declares them.
CONTRACT = {
    "input_ids": ((1, 1), np.int32),
    "position_ids": ((1,), np.int32),
    "attention_mask": ((1, 1, 1, 200), np.float16),
    **{f"k_cache_self_{i}_in": ((6, 1, 64, 199), np.float16) for i in range(4)},
    **{f"v_cache_self_{i}_in": ((6, 1, 199, 64), np.float16) for i in range(4)},
    **{f"k_cache_cross_{i}": ((6, 1, 64, 1500), np.float16) for i in range(4)},
    **{f"v_cache_cross_{i}": ((6, 1, 1500, 64), np.float16) for i in range(4)},
}


def _cross():
    return {name: np.zeros(shape, dtype) for name, (shape, dtype) in CONTRACT.items()
            if "cross" in name}


def test_a_step_supplies_every_input_the_model_declares_with_the_right_shape():
    feeds = step_inputs(50258, 0, initial_state(TINY), _cross(), TINY)

    assert set(feeds) == set(CONTRACT), (
        f"missing {set(CONTRACT) - set(feeds)}, unexpected {set(feeds) - set(CONTRACT)}"
    )
    for name, (shape, dtype) in CONTRACT.items():
        assert feeds[name].shape == shape, f"{name}: {feeds[name].shape} != {shape}"
        assert feeds[name].dtype == dtype, f"{name}: {feeds[name].dtype} != {dtype}"


def test_keys_are_stored_transposed_and_values_are_not():
    """[heads, 1, dim, time] for keys against [heads, 1, time, dim] for values —
    swapping them is silent: the shapes only differ where dim != cache length."""
    state = initial_state(TINY)
    assert state["k_cache_self_0_in"].shape == (6, 1, 64, CACHE_LEN)
    assert state["v_cache_self_0_in"].shape == (6, 1, CACHE_LEN, 64)


def test_the_mask_opens_one_slot_per_step_from_the_right():
    """The cache is right-aligned, so position n reveals index (200 - n - 1)."""
    state = initial_state(TINY)
    assert (state["attention_mask"] == MASK_NEG).all()          # nothing visible yet

    for position in range(5):
        feeds = step_inputs(1, position, state, _cross(), TINY)
        mask = feeds["attention_mask"][0, 0, 0]
        visible = np.flatnonzero(mask == 0.0)
        assert list(visible) == list(range(MAX_DECODE_LEN - position - 1, MAX_DECODE_LEN)), (
            f"at position {position} the visible window is {visible}"
        )
        assert mask[MAX_DECODE_LEN - position - 2] == MASK_NEG   # the next slot stays hidden


def test_the_mask_value_is_the_one_the_hexagon_can_quantize():
    """float32's min in a mask is what collapsed MiniLM's activations to zero;
    Qualcomm's own Whisper uses -100 for the same reason."""
    assert MASK_NEG == -100.0
    assert initial_state(TINY)["attention_mask"].dtype == np.float16
    assert np.isfinite(MASK_NEG)


def test_each_steps_output_cache_becomes_the_next_steps_input():
    state = initial_state(TINY)
    outputs = {f"k_cache_self_{i}_out": np.full((6, 1, 64, CACHE_LEN), i + 1, np.float16)
               for i in range(TINY.layers)}
    outputs.update({f"v_cache_self_{i}_out": np.full((6, 1, CACHE_LEN, 64), i + 1, np.float16)
                    for i in range(TINY.layers)})

    state = advance(state, outputs, TINY)
    for i in range(TINY.layers):
        assert (state[f"k_cache_self_{i}_in"] == i + 1).all()
        assert (state[f"v_cache_self_{i}_in"] == i + 1).all()
    assert (state["attention_mask"] == MASK_NEG).all()          # advance must not touch the mask


def test_shape_is_read_from_the_model_rather_than_assumed(tmp_path):
    """A different Whisper size has different layers and heads; nothing is hardcoded."""
    import onnx
    from onnx import TensorProto, helper

    def value(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape)

    inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT32, [1, 1]),
              value("attention_mask", [1, 1, 1, 448])]
    for layer in range(6):                                   # pretend: 6 layers, 8 heads, 64 dim
        inputs += [value(f"k_cache_self_{layer}_in", [8, 1, 64, 447]),
                   value(f"v_cache_self_{layer}_in", [8, 1, 447, 64])]
    graph = helper.make_graph([helper.make_node("Identity", ["input_ids"], ["out"])], "d",
                              inputs, [helper.make_tensor_value_info("out", TensorProto.INT32, [1, 1])])
    path = tmp_path / "decoder.onnx"
    onnx.save(helper.make_model(graph, ir_version=10), path)

    shape = DecoderShape.from_model(path)
    assert (shape.layers, shape.heads, shape.head_dim) == (6, 8, 64)
    assert (shape.cache_len, shape.max_decode_len) == (447, 448)


@pytest.mark.skipif(True, reason="needs a physical Snapdragon device: EPContext graph, QNN EP only")
def test_end_to_end_decode_on_the_npu():
    """Kept visible on purpose — this is the one claim a cloud device cannot settle."""
