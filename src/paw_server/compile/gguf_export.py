"""GGUF LoRA export for llama.cpp, mirroring the authors' conversion script."""

from pathlib import Path

import torch

# Mapper module name -> llama.cpp tensor stem, per interpreter arch.
MODULE_TO_GGUF = {
    "qwen3": {
        "q_proj": "attn_q",
        "k_proj": "attn_k",
        "v_proj": "attn_v",
        "o_proj": "attn_output",
        "gate_proj": "ffn_gate",
        "up_proj": "ffn_up",
        "down_proj": "ffn_down",
    },
    "gpt2": {
        "c_attn": "attn_qkv",
        "attn_c_proj": "attn_output",
        "c_fc": "ffn_up",
        "mlp_c_proj": "ffn_down",
    },
}


def write_gguf_adapter(
    lora_params: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]],
    output_path: str | Path,
    lora_alpha: float,
    arch: str = "qwen3",
    quant: str = "Q4_0",
) -> None:
    """quant: "Q4_0" (production, ~22 MB), or "F32" (~154 MB, lossless)."""
    import gguf

    def _tensor_data(t: torch.Tensor):
        data = t.detach().float().cpu().numpy()
        if quant == "F32":
            return data, None
        qtype = gguf.GGMLQuantizationType[quant]
        return gguf.quants.quantize(data, qtype), qtype

    module_map = MODULE_TO_GGUF[arch]
    writer = gguf.GGUFWriter(str(output_path), arch=arch)
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", lora_alpha)

    for (layer_idx, mod_name), (a, b) in sorted(lora_params.items()):
        gguf_name = module_map[mod_name]
        base_name = f"blk.{layer_idx}.{gguf_name}.weight"
        for suffix, tensor in ((".lora_a", a), (".lora_b", b)):
            data, qtype = _tensor_data(tensor)
            # With raw_dtype set, gguf derives the logical shape from the
            # quantized array's byte shape — do not pass raw_shape.
            writer.add_tensor(f"{base_name}{suffix}", data, raw_dtype=qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
