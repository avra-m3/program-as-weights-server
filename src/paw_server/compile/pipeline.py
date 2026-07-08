"""Local PAW compile: spec -> pseudo-program + LoRA adapter.

Pipeline (mirrors the hosted compile, per the paper and research code):
1. The untrained pseudo compiler (off-the-shelf Qwen3-4B-Instruct-2507)
   generates a pseudo-program from the spec (chat-templated "examples"
   prompt, greedy decoding). The trained compiler cannot do this step:
   its own generations degenerate (it is an encoder after training).
2. One forward pass through the *trained* compiler over
   [chat(minimal(spec))] [pseudo-program] [EOS] [<prefix_1>..<prefix_64>]
   with hidden states captured at the 64 prefix positions from 28
   depth-ratio-aligned compiler layers.
3. The published mapper turns those hidden states into 28x7 LoRA A/B
   matrices (mixture of shared bases).
4. The result is exported as a standard PEFT adapter plus the program
   artifact files used by the official runtime (prompt_template.txt,
   meta.json), matching the layout of programasweights/paw-programs.
"""

import datetime
import gc
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from paw_server.compile.mapper import depth_ratio_layers, load_mapper
from paw_server.compile.prompts import compiler_prompt, interpreter_prompt

COMPILER_REPO = "programasweights/paw-4b-qwen3-0.6b"
PSEUDO_COMPILER_REPO = "Qwen/Qwen3-4B-Instruct-2507"
PREFIX_STEPS = 64

# PEFT module name -> submodule path inside a Qwen3 decoder layer.
MODULE_PARENT = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def interpreter_module_dims(config) -> dict[str, tuple[int, int]]:
    """(d_in, d_out) of each LoRA target module, from the interpreter config."""
    hidden = config.hidden_size
    head_dim = getattr(config, "head_dim", hidden // config.num_attention_heads)
    q_out = config.num_attention_heads * head_dim
    kv_out = config.num_key_value_heads * head_dim
    inter = config.intermediate_size
    return {
        "q_proj": (hidden, q_out),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
        "o_proj": (q_out, hidden),
        "gate_proj": (hidden, inter),
        "up_proj": (hidden, inter),
        "down_proj": (inter, hidden),
    }


def _render_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def _prefix_token_ids(tokenizer, embedding_rows: int) -> list[int]:
    tokens = [f"<prefix_{i}>" for i in range(1, PREFIX_STEPS + 1)]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if any(i is None or i < 0 for i in ids):
        added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        ids = tokenizer.convert_tokens_to_ids(tokens)
        if added and any(i >= embedding_rows for i in ids):
            raise RuntimeError(
                "Prefix tokens were assigned ids beyond the compiler's embedding "
                f"table ({embedding_rows} rows); tokenizer/weights mismatch."
            )
    return ids


def _cuda_max_memory() -> dict[int | str, int]:
    """Explicit `max_memory` (in raw bytes) for a single-GPU box that can't
    necessarily fit a whole 4B model.

    We *want* transformers/accelerate's auto-balancer to spill onto CPU when
    the GPU is tight (`device_map="auto"`), but on this pinned
    transformers/accelerate pair (5.13.0 / 1.14.0), letting them
    *auto-detect* memory blows up: `_get_device_map` treats any string
    `device_map` as an auto-balance request and calls
    `get_balanced_memory -> get_max_memory`, which raises
    `ValueError: size ... is not in a valid format` because the
    auto-detected value it feeds into `convert_file_size_to_int` is a
    bare digit string (no unit suffix) rather than an int or a
    "<N><unit>" string. Passing `max_memory` ourselves (as plain ints,
    which `convert_file_size_to_int` accepts directly) skips that broken
    auto-detection path entirely while still getting real GPU/CPU
    balancing out of `device_map="auto"`.
    """
    import psutil

    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    # Mirror accelerate's own single-GPU rule of thumb: keep ~10% headroom
    # for activations/buffers rather than packing the GPU to the brim.
    gpu_budget = int(free_bytes * 0.9)
    cpu_budget = int(psutil.virtual_memory().available * 0.9)
    return {0: gpu_budget, "cpu": cpu_budget}


def _load_model(repo: str, device: str, subfolder: str | None = None):
    """Load a causal LM with shard streaming straight to the target device.

    Loading to CPU and then .to(device) doubles peak memory (~16 GB for the
    4B models) and can push a 24 GB machine deep into swap.

    On CUDA we use ``device_map="auto"`` with a 1 GiB headroom so that
    accelerate can offload layers to CPU when the model is too large for
    the GPU.  Without the headroom the 4B pseudo compiler (~7.5 GiB in
    bf16) can OOM during weight-materialization on 8 GiB cards because
    intermediate copies briefly push memory past the limit.
    """
    kwargs: dict = {"dtype": torch.bfloat16, "device_map": device}
    if device == "cuda":
        # See _cuda_max_memory: request real auto-balancing (GPU can be too
        # small for a 4B model) but hand it explicit, valid max_memory so we
        # don't hit the accelerate auto-detection bug.
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = _cuda_max_memory()
    if subfolder:
        kwargs["subfolder"] = subfolder

    model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    return model.eval()


def _input_device(model, device: str) -> torch.device:
    """The device that top-level inputs (input_ids) must live on.

    With ``device_map="auto"`` accelerate shards the model across GPU and
    CPU and installs hooks that move each submodule's inputs to that
    submodule's device -- but only *after* the top-level tensor reaches the
    first module. So ``input_ids`` must start on the device of the module
    that first consumes them (the input embedding). That is almost always
    GPU 0, but if ``max_memory[0]`` is tight accelerate can place the
    embedding on CPU, in which case hardcoding ``"cuda"`` would raise a
    device-mismatch. Derive it from the actual placement instead.
    """
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map:
        # Find where the input-embedding module was placed. Prefer looking it
        # up by the embedding's own module name rather than trusting dict
        # order; fall back to the first entry (accelerate lists modules in
        # traversal order, so the embedding is normally first anyway).
        emb = model.get_input_embeddings()
        emb_key = next(
            (name for name, mod in model.named_modules() if mod is emb), None
        )
        placed = None
        if emb_key is not None:
            # hf_device_map keys are module prefixes ("model.embed_tokens" or
            # a parent like "model"); match the longest prefix of emb_key.
            for key in sorted(hf_map, key=len, reverse=True):
                if emb_key == key or emb_key.startswith(key + "."):
                    placed = hf_map[key]
                    break
        if placed is None:
            placed = next(iter(hf_map.values()))
        # accelerate normalises device values to ints (GPU index), "cpu",
        # or "disk".
        if isinstance(placed, int):
            return torch.device("cuda", placed)
        if placed in ("cpu", "disk"):
            return torch.device("cpu")
        return torch.device(placed)
    # No sharding (single-device load): inputs go on the model's device.
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(device)


def _free_model(model, device: str) -> None:
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def compile_spec(
    spec: str,
    out_dir: str | Path,
    pseudo_style: str = "examples",
    max_new_tokens: int = 512,
    device: str | None = None,
    write_gguf: bool = False,
) -> Path:
    """Compile `spec` into a PAW program directory. Returns the directory."""
    device = device or pick_device()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta_path = hf_hub_download(COMPILER_REPO, "meta.json")
    meta = json.loads(Path(meta_path).read_text())
    interpreter_id = meta["interpreter_model"]
    lora_rank = meta["lora_rank"]
    lora_alpha = meta["lora_alpha"]
    num_bases = meta["lora_num_bases"]
    assert meta["prefix_steps"] == PREFIX_STEPS

    int_config = AutoConfig.from_pretrained(interpreter_id)
    module_dims = interpreter_module_dims(int_config)
    num_student_layers = int_config.num_hidden_layers

    # --- 1. Generate the pseudo-program with the *untrained* pseudo compiler
    # (paper §3.1: "an off-the-shelf Qwen3-4B-Instruct-2507 model that we
    # never train"). The trained compiler checkpoint cannot generate — its
    # LM ability collapsed during training; it only encodes (verified: its
    # greedy output degenerates on any prompt, on every device/dtype).
    print(f"Loading pseudo compiler ({PSEUDO_COMPILER_REPO}) on {device} ...")
    t0 = time.perf_counter()
    ps_tokenizer = AutoTokenizer.from_pretrained(PSEUDO_COMPILER_REPO)
    ps_model = _load_model(PSEUDO_COMPILER_REPO, device)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    gen_prompt = _render_chat(ps_tokenizer, compiler_prompt(spec, style=pseudo_style))
    gen_inputs = ps_tokenizer(gen_prompt, return_tensors="pt").to(
        _input_device(ps_model, device)
    )
    t0 = time.perf_counter()
    with torch.no_grad():
        gen_out = ps_model.generate(
            **gen_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=ps_tokenizer.pad_token_id or ps_tokenizer.eos_token_id,
        )
    new_tokens = gen_out[0, gen_inputs["input_ids"].shape[1] :]
    pseudo_program = ps_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print(
        f"  pseudo-program: {len(new_tokens)} tokens "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    _free_model(ps_model, device)

    # --- 2. Prefix-hidden forward pass through the *trained* compiler.
    # Sequence: [chat(minimal(spec))] [pseudo] [EOS] [prefix tokens]; the
    # "minimal" prompt is always used here regardless of pseudo_style,
    # matching training (meta.json compiler_prompt_style="minimal").
    print(f"Loading trained compiler ({COMPILER_REPO}) on {device} ...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(COMPILER_REPO, subfolder="compiler")
    model = _load_model(COMPILER_REPO, device, subfolder="compiler")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    embedding_rows = model.get_input_embeddings().weight.shape[0]
    prefix_ids = _prefix_token_ids(tokenizer, embedding_rows)

    hidden_prompt = _render_chat(tokenizer, compiler_prompt(spec, style="minimal"))
    prompt_ids = tokenizer(hidden_prompt)["input_ids"]
    pseudo_ids = tokenizer(pseudo_program, add_special_tokens=False)["input_ids"]
    full_ids = prompt_ids + pseudo_ids + [tokenizer.eos_token_id] + prefix_ids
    input_ids = torch.tensor([full_ids], device=_input_device(model, device))

    t0 = time.perf_counter()
    with torch.no_grad():
        fwd = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = fwd.hidden_states  # embeddings + one per compiler layer
    num_teacher_layers = len(hidden_states) - 1
    teacher_layers = depth_ratio_layers(num_teacher_layers, num_student_layers)
    prefix_hidden = [
        hidden_states[t + 1][0, -PREFIX_STEPS:, :].float().cpu() for t in teacher_layers
    ]
    print(f"  prefix hidden states extracted in {time.perf_counter() - t0:.1f}s")

    del fwd, hidden_states
    _free_model(model, device)

    # --- 3. Map hidden states to LoRA A/B.
    mapper_path = hf_hub_download(COMPILER_REPO, "lora_mapper.pt")
    mapper = load_mapper(
        mapper_path,
        teacher_hidden_size=prefix_hidden[0].shape[-1],
        student_num_layers=num_student_layers,
        module_dims=module_dims,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        num_bases=num_bases,
    )
    lora_params = mapper(prefix_hidden)

    # --- 4. Export: PEFT adapter + runtime artifact files.
    tensors = {}
    for (layer, module), (a, b) in lora_params.items():
        base = f"base_model.model.model.layers.{layer}.{MODULE_PARENT[module]}.{module}"
        tensors[f"{base}.lora_A.weight"] = (
            a.detach().to("cpu", torch.bfloat16).contiguous()
        )
        tensors[f"{base}.lora_B.weight"] = (
            b.detach().to("cpu", torch.bfloat16).contiguous()
        )
    save_file(tensors, str(out / "adapter_model.safetensors"))

    adapter_config = {
        "peft_type": "LORA",
        "base_model_name_or_path": interpreter_id,
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": sorted(module_dims.keys()),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    (out / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))

    (out / "pseudo_program.txt").write_text(pseudo_program + "\n")

    int_tokenizer = AutoTokenizer.from_pretrained(interpreter_id)
    template = int_tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": interpreter_prompt(pseudo_program, "{INPUT_PLACEHOLDER}"),
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    (out / "prompt_template.txt").write_text(template)

    (out / "meta.json").write_text(
        json.dumps(
            {
                "version": 3,
                "program_id": out.name,
                "spec": spec,
                "compiler_snapshot": "paw-4b-qwen3-0.6b-20260407",
                "compiler_fingerprint": "local-reimplementation",
                "interpreter": interpreter_id,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "prefix_steps": None,
                "created_at": datetime.datetime.now(datetime.UTC).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            },
            indent=2,
        )
    )

    if write_gguf:
        from paw_server.compile.gguf_export import write_gguf_adapter

        write_gguf_adapter(lora_params, out / "adapter.gguf", lora_alpha=lora_alpha)

    print(f"Compiled program written to {out}")
    return out
