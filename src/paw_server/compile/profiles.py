"""Per-compiler profiles for the two published PAW compilers.

Both compilers share the same trained Qwen3-4B compiler architecture and
LoRA mapper design; they differ only in the *interpreter* they target, so
module naming, PEFT/GGUF tensor paths and interpreter-prompt templating
vary per compiler. Every value here mirrors the published artifacts:

- mapper module names come from the lora_mapper.pt state-dict keys
  (for GPT-2: attn_c_proj / c_attn / c_fc / mlp_c_proj, disambiguating
  the two c_proj Conv1D modules),
- PEFT tensor paths, adapter_config target_modules, GGUF arch/tensor
  names and the prompt_template.txt shape were checked against official
  programasweights/paw-programs artifacts (Qwen3 templates are
  chat-templated; GPT-2 has no chat template and uses the plain
  interpreter prompt),
- meta.json fields (compiler_snapshot, prefix_steps) match what the
  hosted compiler writes for each runtime.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

DEFAULT_COMPILER = "paw-4b-qwen3-0.6b"


def qwen3_module_dims(config) -> dict[str, tuple[int, int]]:
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


def gpt2_module_dims(config) -> dict[str, tuple[int, int]]:
    hidden = config.hidden_size  # n_embd
    inter = getattr(config, "n_inner", None) or 4 * hidden
    return {
        "c_attn": (hidden, 3 * hidden),
        "attn_c_proj": (hidden, hidden),
        "c_fc": (hidden, inter),
        "mlp_c_proj": (inter, hidden),
    }


@dataclass(frozen=True)
class CompilerProfile:
    name: str  # public compiler name (REST API)
    repo: str  # HF repo: compiler/ subfolder + lora_mapper.pt + meta.json
    snapshot: str  # compiler_snapshot written into program meta.json
    interpreter: str  # HF interpreter model id (matches meta interpreter_model)
    runtime_id: str  # SDK runtime manifest id for compiled programs
    gguf_arch: str  # llama.cpp arch string for adapter.gguf
    peft_layer_fmt: str  # PEFT path of decoder layer {layer}
    peft_modules: dict[str, str]  # mapper module -> path inside a layer
    chat_template: bool  # wrap interpreter prompt in tokenizer chat template?
    meta_prefix_steps: int | None  # prefix_steps value in program meta.json
    module_dims: Callable[..., dict[str, tuple[int, int]]] = field(repr=False)

    def peft_key(self, layer: int, module: str) -> str:
        return f"{self.peft_layer_fmt.format(layer=layer)}.{self.peft_modules[module]}"


QWEN3_06B = CompilerProfile(
    name="paw-4b-qwen3-0.6b",
    repo="programasweights/paw-4b-qwen3-0.6b",
    snapshot="paw-4b-qwen3-0.6b-20260407",
    interpreter="Qwen/Qwen3-0.6B",
    runtime_id="qwen3-0.6b-q6_k",
    gguf_arch="qwen3",
    peft_layer_fmt="base_model.model.model.layers.{layer}",
    peft_modules={
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "gate_proj": "mlp.gate_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
    },
    chat_template=True,
    meta_prefix_steps=None,
    module_dims=qwen3_module_dims,
)

GPT2 = CompilerProfile(
    name="paw-4b-gpt2",
    repo="programasweights/paw-4b-gpt2",
    snapshot="paw-4b-gpt2-20260406",
    interpreter="gpt2",
    runtime_id="gpt2-q8_0",
    gguf_arch="gpt2",
    peft_layer_fmt="base_model.model.transformer.h.{layer}",
    peft_modules={
        "c_attn": "attn.c_attn",
        "attn_c_proj": "attn.c_proj",
        "c_fc": "mlp.c_fc",
        "mlp_c_proj": "mlp.c_proj",
    },
    chat_template=False,
    meta_prefix_steps=64,
    module_dims=gpt2_module_dims,
)

PROFILES: dict[str, CompilerProfile] = {p.name: p for p in (QWEN3_06B, GPT2)}


def get_profile(compiler: str) -> CompilerProfile:
    """Resolve a compiler name or snapshot id to its profile."""
    for profile in PROFILES.values():
        if compiler in (profile.name, profile.snapshot):
            return profile
    raise KeyError(f"Unknown compiler '{compiler}'.")
