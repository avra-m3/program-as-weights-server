"""Unit tests for the local PAW compile pipeline (no model downloads)."""

import torch

from paw_server.compile.mapper import LoraMapper, depth_ratio_layers
from paw_server.compile.pipeline import MODULE_PARENT, interpreter_module_dims
from paw_server.compile.prompts import compiler_prompt, interpreter_prompt

QWEN3_06B_DIMS = {
    "q_proj": (1024, 2048),
    "k_proj": (1024, 1024),
    "v_proj": (1024, 1024),
    "o_proj": (2048, 1024),
    "gate_proj": (1024, 3072),
    "up_proj": (1024, 3072),
    "down_proj": (3072, 1024),
}


def test_depth_ratio_matches_reference_rule():
    # Reference: teacher_layer_idx = int((l+1) * ratio) - 1, clamped.
    layers = depth_ratio_layers(36, 28)
    assert len(layers) == 28
    assert layers[0] == 0
    assert layers[-1] == 35
    assert layers == sorted(layers)
    # Identity mapping when teacher == student.
    assert depth_ratio_layers(28, 28) == list(range(28))


def test_mapper_state_dict_shapes_match_published_checkpoint():
    """Constructed mapper must accept the published lora_mapper.pt layout."""
    mapper = LoraMapper(
        teacher_hidden_size=2560,
        student_num_layers=28,
        module_dims=QWEN3_06B_DIMS,
        lora_rank=64,
        lora_alpha=16.0,
        num_bases=64,
    )
    sd = mapper.state_dict()
    # Shapes observed in programasweights/paw-4b-qwen3-0.6b/lora_mapper.pt.
    assert sd["trunk.0.weight"].shape == (2560, 2560)
    assert sd["coeff_head.weight"].shape == (25088, 2560)
    assert sd["A_bases_q_proj"].shape == (64, 64, 1024)
    assert sd["B_bases_q_proj"].shape == (64, 2048, 64)
    assert sd["A_bases_down_proj"].shape == (64, 64, 3072)
    assert sd["B_bases_down_proj"].shape == (64, 1024, 64)
    assert len(sd) == 4 + 14  # trunk w/b, head w/b, 7 modules x (A, B)


def test_mapper_forward_output_shapes_and_scaling():
    mapper = LoraMapper(
        teacher_hidden_size=32,
        student_num_layers=3,
        module_dims={"q_proj": (16, 24), "down_proj": (48, 16)},
        lora_rank=4,
        lora_alpha=16.0,
        num_bases=5,
    )
    hidden = [torch.randn(7, 32) for _ in range(3)]
    lora = mapper(hidden)
    expected_keys = {(i, m) for i in range(3) for m in ("q_proj", "down_proj")}
    assert set(lora) == expected_keys
    a, b = lora[(1, "q_proj")]
    assert a.shape == (4, 16) and b.shape == (24, 4)
    assert mapper.lora_scaling == 4.0


def test_mapper_coefficients_mix_bases():
    """A/B must be linear mixtures of the bases driven by the coeff head."""
    torch.manual_seed(0)
    mapper = LoraMapper(
        teacher_hidden_size=8,
        student_num_layers=1,
        module_dims={"q_proj": (4, 4)},
        lora_rank=2,
        lora_alpha=16.0,
        num_bases=3,
    )
    with torch.no_grad():
        mapper.A_bases_q_proj.normal_()
        mapper.B_bases_q_proj.normal_()
        mapper.coeff_head.weight.normal_()
    hidden = [torch.randn(5, 8)]
    lora = mapper(hidden)
    a, _ = lora[(0, "q_proj")]

    # Recompute by hand.
    h = torch.stack(hidden).mean(0).mean(0)
    z = mapper.trunk(h)
    coeffs = mapper.coeff_head(z).view(1, 1, 3, 2)
    expected_a = torch.einsum("n,nrd->rd", coeffs[0, 0, :, 0], mapper.A_bases_q_proj)
    torch.testing.assert_close(a, expected_a)


def test_interpreter_module_dims_qwen3_06b():
    class Cfg:
        hidden_size = 1024
        num_attention_heads = 16
        num_key_value_heads = 8
        head_dim = 128
        intermediate_size = 3072

    assert interpreter_module_dims(Cfg()) == QWEN3_06B_DIMS
    assert set(QWEN3_06B_DIMS) == set(MODULE_PARENT)


def test_gguf_adapter_roundtrip(tmp_path):
    import gguf

    from paw_server.compile.gguf_export import write_gguf_adapter

    lora = {
        (0, "q_proj"): (torch.randn(64, 1024), torch.randn(2048, 64)),
        (1, "down_proj"): (torch.randn(64, 3072), torch.randn(1024, 64)),
    }
    path = tmp_path / "adapter.gguf"
    write_gguf_adapter(lora, path, lora_alpha=16.0, quant="Q4_0")

    reader = gguf.GGUFReader(str(path))
    tensors = {t.name: t for t in reader.tensors}
    assert set(tensors) == {
        "blk.0.attn_q.weight.lora_a",
        "blk.0.attn_q.weight.lora_b",
        "blk.1.ffn_down.weight.lora_a",
        "blk.1.ffn_down.weight.lora_b",
    }
    # GGUF stores shapes reversed (ggml order); logical (64, 1024) -> [1024, 64].
    assert list(tensors["blk.0.attn_q.weight.lora_a"].shape) == [1024, 64]
    assert list(tensors["blk.1.ffn_down.weight.lora_b"].shape) == [64, 1024]
    assert all(t.tensor_type.name == "Q4_0" for t in tensors.values())


def test_prompts_match_paper_appendix_c():
    minimal = compiler_prompt("count words", style="minimal")
    assert minimal == "[SPEC]\ncount words\n[END_SPEC]\n\n[PSEUDO_PROGRAM]"

    examples = compiler_prompt("count words", style="examples")
    assert examples.startswith("You are PAW-Compiler.")
    assert "[END_PSEUDO_PROGRAM]" in examples

    interp = interpreter_prompt("Task: count words.", "hello world")
    assert interp == "Task: count words.\n\n[INPUT]\nhello world\n[END_INPUT]"
