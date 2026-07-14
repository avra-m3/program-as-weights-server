"""Unit tests for the local PAW compile pipeline (no model downloads)."""

import torch

from paw_server.compile.mapper import LoraMapper, depth_ratio_layers
from paw_server.compile.pipeline import (
    _DEFAULT_CUDA_RUNTIME_RESERVE_GB,
    MODULE_PARENT,
    PREFIX_STEPS,
    _capture_prefix_hidden,
    _cuda_runtime_reserve_bytes,
    _input_device,
    interpreter_module_dims,
)
from paw_server.compile.profiles import GPT2, QWEN3_06B, get_profile
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

# Shapes observed in programasweights/paw-4b-gpt2/lora_mapper.pt.
GPT2_DIMS = {
    "c_attn": (768, 2304),
    "attn_c_proj": (768, 768),
    "c_fc": (768, 3072),
    "mlp_c_proj": (3072, 768),
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
    assert QWEN3_06B.module_dims(Cfg()) == QWEN3_06B_DIMS


def test_gpt2_module_dims():
    class Cfg:
        hidden_size = 768
        n_inner = None  # GPT2Config default: 4 * n_embd

    assert GPT2.module_dims(Cfg()) == GPT2_DIMS
    assert set(GPT2_DIMS) == set(GPT2.peft_modules)


def test_gpt2_mapper_state_dict_shapes_match_published_checkpoint():
    """Constructed mapper must accept the paw-4b-gpt2 lora_mapper.pt layout."""
    mapper = LoraMapper(
        teacher_hidden_size=2560,
        student_num_layers=12,
        module_dims=GPT2_DIMS,
        lora_rank=64,
        lora_alpha=16.0,
        num_bases=64,
    )
    sd = mapper.state_dict()
    assert sd["coeff_head.weight"].shape == (6144, 2560)  # 12 * 4 * 64 * 2
    assert sd["A_bases_c_attn"].shape == (64, 64, 768)
    assert sd["B_bases_c_attn"].shape == (64, 2304, 64)
    assert sd["A_bases_mlp_c_proj"].shape == (64, 64, 3072)
    assert sd["B_bases_mlp_c_proj"].shape == (64, 768, 64)
    assert len(sd) == 4 + 8  # trunk w/b, head w/b, 4 modules x (A, B)


def test_profile_peft_keys_match_official_artifacts():
    # Verified against programasweights/paw-programs adapter tensors.
    assert (
        QWEN3_06B.peft_key(3, "q_proj")
        == "base_model.model.model.layers.3.self_attn.q_proj"
    )
    assert GPT2.peft_key(0, "c_attn") == "base_model.model.transformer.h.0.attn.c_attn"
    assert (
        GPT2.peft_key(11, "mlp_c_proj")
        == "base_model.model.transformer.h.11.mlp.c_proj"
    )


def test_profile_resolution_by_name_and_snapshot():
    assert get_profile("paw-4b-gpt2") is GPT2
    assert get_profile("paw-4b-gpt2-20260406") is GPT2  # snapshot id
    assert get_profile("paw-4b-qwen3-0.6b") is QWEN3_06B


def test_gpt2_depth_ratio_every_third_layer():
    # GPT-2: 12 student layers against the 36-layer compiler = every 3rd.
    assert depth_ratio_layers(36, 12) == list(range(2, 36, 3))


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


def test_gguf_adapter_gpt2_naming(tmp_path):
    """Tensor names/arch verified against an official gpt2 adapter.gguf."""
    import gguf

    from paw_server.compile.gguf_export import write_gguf_adapter

    lora = {
        (0, "c_attn"): (torch.randn(64, 768), torch.randn(2304, 64)),
        (0, "attn_c_proj"): (torch.randn(64, 768), torch.randn(768, 64)),
        (11, "c_fc"): (torch.randn(64, 768), torch.randn(3072, 64)),
        (11, "mlp_c_proj"): (torch.randn(64, 3072), torch.randn(768, 64)),
    }
    path = tmp_path / "adapter.gguf"
    write_gguf_adapter(lora, path, lora_alpha=16.0, arch="gpt2")

    reader = gguf.GGUFReader(str(path))
    arch = reader.fields["general.architecture"]
    assert arch.parts[arch.data[0]].tobytes() == b"gpt2"
    names = {t.name for t in reader.tensors}
    assert names == {
        "blk.0.attn_qkv.weight.lora_a",
        "blk.0.attn_qkv.weight.lora_b",
        "blk.0.attn_output.weight.lora_a",
        "blk.0.attn_output.weight.lora_b",
        "blk.11.ffn_up.weight.lora_a",
        "blk.11.ffn_up.weight.lora_b",
        "blk.11.ffn_down.weight.lora_a",
        "blk.11.ffn_down.weight.lora_b",
    }


class _FakeModel:
    """Minimal stand-in exposing the attributes _input_device inspects."""

    def __init__(self, hf_device_map=None, param_device="cpu"):
        self._emb = torch.nn.Embedding(4, 4)
        self._param_device = torch.device(param_device)
        if hf_device_map is not None:
            self.hf_device_map = hf_device_map

    def get_input_embeddings(self):
        return self._emb

    def named_modules(self):
        # The embedding lives at "model.embed_tokens", matching a real
        # Qwen3 causal LM's module tree.
        yield "", self
        yield "model", torch.nn.Module()
        yield "model.embed_tokens", self._emb

    def parameters(self):
        p = torch.nn.Parameter(torch.zeros(1, device=self._param_device))
        yield p


def test_input_device_no_sharding_uses_param_device():
    # No hf_device_map => single-device load; follow the model's params.
    model = _FakeModel(hf_device_map=None, param_device="cpu")
    assert _input_device(model, "cuda") == torch.device("cpu")


def test_input_device_sharded_embedding_on_gpu():
    # device_map="auto" placement with the embedding on GPU 0.
    model = _FakeModel(
        hf_device_map={"model.embed_tokens": 0, "model.layers.20": "cpu"}
    )
    assert _input_device(model, "cuda") == torch.device("cuda", 0)


def test_input_device_sharded_embedding_offloaded_to_cpu():
    # The edge case the fix targets: a tight GPU pushes the embedding to CPU,
    # so inputs must be on CPU (hardcoding "cuda" would device-mismatch).
    model = _FakeModel(hf_device_map={"model.embed_tokens": "cpu", "model.layers.0": 0})
    assert _input_device(model, "cuda") == torch.device("cpu")


def test_input_device_sharded_via_parent_prefix():
    # accelerate may key the map by a parent module ("model") rather than the
    # embedding leaf; the longest-prefix match must still resolve it.
    model = _FakeModel(hf_device_map={"model": 0})
    assert _input_device(model, "cuda") == torch.device("cuda", 0)


def test_cuda_runtime_reserve_default(monkeypatch):
    monkeypatch.delenv("PAW_CUDA_RUNTIME_RESERVE_GB", raising=False)
    assert _cuda_runtime_reserve_bytes() == int(
        _DEFAULT_CUDA_RUNTIME_RESERVE_GB * 1024**3
    )


def test_cuda_runtime_reserve_env_override(monkeypatch):
    monkeypatch.setenv("PAW_CUDA_RUNTIME_RESERVE_GB", "3")
    assert _cuda_runtime_reserve_bytes() == 3 * 1024**3


def test_cuda_runtime_reserve_blank_env_falls_back_to_default(monkeypatch):
    # An empty string must not be parsed as 0 (which would pack the GPU).
    monkeypatch.setenv("PAW_CUDA_RUNTIME_RESERVE_GB", "")
    assert _cuda_runtime_reserve_bytes() == int(
        _DEFAULT_CUDA_RUNTIME_RESERVE_GB * 1024**3
    )


def _tiny_qwen3():
    """A tiny, randomly-initialised Qwen3 causal LM (no download)."""
    import torch as _torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    _torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    return Qwen3ForCausalLM(cfg).eval()


def test_capture_prefix_hidden_matches_output_hidden_states():
    """The hook-based capture must reproduce output_hidden_states EXACTLY.

    Regression guard for the bug where the final teacher layer was captured
    PRE-final-norm (raw decoder output) instead of POST-norm. transformers
    ties hidden_states[-1] to last_hidden_state (post model.norm), and the
    mapper was trained on that; feeding the pre-norm residual corrupts the
    LoRA and makes the interpreter emit a single repeated token. Because
    depth_ratio_layers always maps the last student layer onto the final
    teacher layer, this must match at every slot -- especially the last.
    """
    model = _tiny_qwen3()
    num_teacher = model.config.num_hidden_layers
    ids = torch.randint(0, 64, (1, PREFIX_STEPS + 4))

    # Include the final layer (the previously-broken case) plus a duplicate,
    # mirroring how depth_ratio_layers can repeat teacher indices.
    teacher_layers = [0, 1, num_teacher - 1, num_teacher - 1]

    with torch.no_grad():
        out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    reference = [
        out.hidden_states[t + 1][0, -PREFIX_STEPS:, :].float().cpu()
        for t in teacher_layers
    ]

    captured = _capture_prefix_hidden(model, ids, teacher_layers)

    assert len(captured) == len(teacher_layers)
    for slot, (ref, got) in enumerate(zip(reference, captured, strict=True)):
        assert torch.allclose(ref, got, atol=1e-5), (
            f"slot {slot} (teacher layer {teacher_layers[slot]}) diverges from "
            f"output_hidden_states; max diff "
            f"{(ref - got).abs().max().item():.3e}"
        )

    # And prove the fix matters: the RAW final-layer output (pre-norm) must
    # NOT match -- otherwise the test could pass even with the bug present.
    raw_final = out.hidden_states  # tuple; index num_teacher is post-norm
    with torch.no_grad():
        pre_norm_final = None

        def _grab(_m, _i, o):
            nonlocal pre_norm_final
            hs = o[0] if isinstance(o, tuple) else o
            pre_norm_final = hs[0, -PREFIX_STEPS:, :].float().cpu()

        h = model.model.layers[num_teacher - 1].register_forward_hook(_grab)
        try:
            model.model(input_ids=ids, use_cache=False)
        finally:
            h.remove()
    assert not torch.allclose(
        raw_final[num_teacher][0, -PREFIX_STEPS:, :].float().cpu(),
        pre_norm_final,
        atol=1e-5,
    ), "pre/post final-norm are identical; test cannot detect the regression"


def test_prompts_match_paper_appendix_c():
    minimal = compiler_prompt("count words", style="minimal")
    assert minimal == "[SPEC]\ncount words\n[END_SPEC]\n\n[PSEUDO_PROGRAM]"

    examples = compiler_prompt("count words", style="examples")
    assert examples.startswith("You are PAW-Compiler.")
    assert "[END_PSEUDO_PROGRAM]" in examples

    interp = interpreter_prompt("Task: count words.", "hello world")
    assert interp == "Task: count words.\n\n[INPUT]\nhello world\n[END_INPUT]"
