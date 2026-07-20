# How PAW compilation works (and what it takes to scale it up)

This documents the *exact* mechanism implemented in
`src/paw_server/compile/`, reverse-validated against the paper
(arXiv:2607.02512), the authors' research code
(anonymous.4open.science/r/programasweights), the published weights (HF
`programasweights/paw-4b-qwen3-0.6b`), and a production program artifact
(HF `programasweights/paw-programs`).

## 1. The cast of models

| Role | Model | Trained? | Size |
|---|---|---|---|
| Pseudo compiler `C_p` | `Qwen/Qwen3-4B-Instruct-2507` (stock) | **never trained** | 4B |
| LoRA compiler `C_L` | fine-tuned Qwen3-4B-Instruct-2507 (`compiler/` on HF) | yes (SFT, eq. 4) | 4B |
| LoRA mapper | `lora_mapper.pt` (652 MB, fp32) | yes | ~163M params |
| Interpreter | `Qwen/Qwen3-0.6B` | **frozen, never trained** | 0.6B |

A compiled "program" = a **pseudo-program** (plain text) + a **LoRA
adapter** (28 layers × 7 modules × rank-64 A/B pairs ≈ 38.5M params,
~77 MB bf16, ~22 MB Q4_0-GGUF).

There is also a **compact variant**, `programasweights/paw-4b-gpt2`
(select it with `"compiler": "paw-4b-gpt2"`): same 4B compiler pair and
mapper design, but the frozen interpreter is a 124M GPT-2. Per-compiler
differences all live in `paw_server/compile/profiles.py` and are mirrored
from that checkpoint's `lora_mapper.pt` and official program artifacts:
12 layers × 4 Conv1D modules (`c_attn`, `attn.c_proj`, `mlp.c_fc`,
`mlp.c_proj` — the mapper names them `c_attn`/`attn_c_proj`/`c_fc`/
`mlp_c_proj` to disambiguate the two `c_proj`s), PEFT paths under
`transformer.h.{i}`, GGUF arch `gpt2`
(`attn_qkv`/`attn_output`/`ffn_up`/`ffn_down`), and a **plain-text**
prompt template (GPT-2 has no chat template). Programs run on the
`gpt2-q8_0` runtime. The compile pipeline is otherwise identical — the
trained compiler is a Qwen3-4B either way, so only the interpreter dims,
tensor naming and templating change.

**GPT-2's 1024-token ceiling is enforced, not worked around.** The
paper's Appendix G training config caps *every* interpreter — Qwen3
0.6B, Qwen3.5 0.8B, and GPT-2 124M alike — at "max interpreter sequence
length 1024"; GPT-2 was simply never asked to exceed a budget the whole
system was already designed around. `CompilerProfile.max_interpreter_tokens`
(`profiles.py`) encodes this per compiler (1024 for GPT-2, matching its
fixed 1024-row learned absolute position-embedding table exactly — there
is no RoPE-style extrapolation available for it; 2048 for Qwen3, which
has genuine RoPE headroom beyond its own training length). It's used in
two places: `RUNTIME_MANIFESTS[...]["local_sdk"]["n_ctx"]` in `app.py`
(previously hardcoded to 2048 for both compilers — a bug for GPT-2, since
that exceeds its real limit), and a compile-time budget check in
`compile_spec` that reserves 256 tokens of headroom for
`[INPUT]`/`[END_INPUT]` plus the (not-yet-known) input and generated
output. If the pseudo-program alone would overflow that budget: a
caller-supplied pseudo-program (`POST /compile/raw`) fails the compile
immediately (there's no text to safely rewrite on the caller's behalf);
a generated one is retried up to twice with an explicit "be shorter"
instruction appended to the compiler prompt before the compile fails.
Programs record `max_interpreter_tokens`, `pseudo_program_tokens`, and
`pseudo_program_regenerated` in `meta.json` for visibility.

An important empirical fact (discovered here, implied by the paper):
the *trained* compiler `C_L` **cannot generate text** — its LM ability
collapsed during training (greedy output degenerates on any prompt, on
every device/dtype). It functions purely as an **encoder**. All text
generation is done by the untrained `C_p`.

## 2. The compile pipeline, step by step

### Step 1 — pseudo-program generation (untrained `C_p`)

The spec is wrapped in the "examples" prompt (verbatim in
`paw_server/compile/prompts.py`, from the paper's Appendix C): *"You are
PAW-Compiler … write a PSEUDO-PROGRAM … Task: + 3-6 example input/output
pairs … under 250 tokens."* This is rendered with Qwen's chat template
(`add_generation_prompt=True, enable_thinking=False`) and decoded
greedily (max 512 new tokens; it stops at EOS on its own, typically
~150-300 tokens). The raw generation, markers included, **is** the
discrete half of the program:

```
[PSEUDO_PROGRAM]
Task: <self-contained restatement>
Examples:
Input: ...
Output: ...
[END_PSEUDO_PROGRAM]
```

### Step 2 — encoding (trained `C_L`, one forward pass)

Build one token sequence:

```
chat_template( "[SPEC]\n{spec}\n[END_SPEC]\n\n[PSEUDO_PROGRAM]" )   ← "minimal" prompt
+ tokens(pseudo_program)
+ [EOS]                                                             ← <|im_end|>
+ [<prefix_1> … <prefix_64>]                                        ← 64 learned tokens
```

The 64 prefix tokens are real vocabulary entries (ids 151669–151732;
the compiler's embedding matrix was extended for them — that's why its
`vocab_size` is 151733). One forward pass with `output_hidden_states=True`,
then hidden states are read **at the 64 prefix positions** from 28 of the
36 compiler layers, aligned by depth ratio:

```
teacher_layer(l) = int((l+1) * 36/28) - 1        for l in 0..27
h_l = hidden_states[teacher_layer(l) + 1][prefix positions]   # skip embeddings
```

### Step 3 — the mapper (hidden states → LoRA weights)

All in fp32 (`lora_mapper.pt`):

1. Stack the 28 tensors of shape (64, 2560), **mean over layers**, then
   **mean over the 64 positions** → one vector `h̄ ∈ R^2560`.
2. Trunk: `z = GELU(Linear_2560×2560(h̄))`.
3. Coefficient head: `Linear_2560→25088(z)`, reshaped to
   `(28 layers, 7 modules, 64 bases, 2)` — modules in *sorted* name
   order, last axis = (A, B).
4. Per (layer, module): mix the shared per-module-type basis matrices:
   `A = Σ_n α_A[n]·A_bases[n]` (64×d_in), `B = Σ_n α_B[n]·B_bases[n]`
   (d_out×64). The bases are trained parameters shared across all 28
   layers of a module type; the ~25k mixing coefficients are all that
   varies per program (the 38.5M-param LoRA has ~25k effective degrees
   of freedom).

### Step 4 — runtime

Attach the LoRA to the frozen Qwen3-0.6B (`out += x @ Aᵀ @ Bᵀ · α/r`,
α/r = 16/64 = 0.25 — standard PEFT semantics, so a stock PEFT adapter
folder is a faithful export). Inference prompt (pre-rendered into
`prompt_template.txt`):

```
<|im_start|>user
{pseudo_program}

[INPUT]
{INPUT_PLACEHOLDER}
[END_INPUT]<|im_end|>
<|im_start|>assistant
<think>

</think>
```

Greedy decode. The interpreter never sees the original spec — the
pseudo-program is its only instruction; the LoRA supplies the rest.

### Measured cost (M4 MacBook Pro, 24 GB, MPS)

| Phase | Cost |
|---|---|
| Load `C_p` (8 GB, shard-streamed) | ~8 s |
| Generate pseudo-program (~190 tok, greedy) | ~60 s |
| Load `C_L` | ~18 s |
| Encoding forward pass + mapper + export | ~11 s |
| **Total compile** | **~95 s** |
| Interpreter inference per call | ~100 ms–1 s |

Peak memory ~8.5 GB (the two 4B models are loaded sequentially, never
both resident). Caveat: on a 24 GB machine, compile competes with
everything else; under heavy swap the generation step degrades ~10×.

## 3. Scaling this up to a larger model

The question splits into three independent axes.

### 3a. Bigger *pseudo compiler* — free, works today

`C_p` is untrained and only writes text; nothing downstream depends on
its identity. You can swap in any stronger instruct model (Qwen3-32B, an
API model, anything) to write better pseudo-programs and keep the
published `C_L` + mapper unchanged. This is the only axis that scales
**without any training**. Expected gains are real but bounded: the
discrete half carries task description + examples, and the paper's
ablations show the LoRA half contributes the larger share of accuracy
(73.8% with LoRA vs. ~52% best fixed-LoRA baseline on FuzzyBench).

### 3b. Bigger *interpreter* — requires retraining, architecture scales cleanly

The compiler/interpreter are a **coupled pair** (paper Appendix N): the
mapper's bases are trained against the exact interpreter geometry. To
move to e.g. Qwen3-1.7B/4B (or any other architecture):

- **Mapper**: purely mechanical growth. Bases per module type are
  `(64, 64, d_in)` / `(64, d_out, 64)`; the coefficient head grows to
  `L·M·64·2`. For Qwen3-1.7B (28 layers, hidden 2048, inter 6144) the
  mapper roughly doubles (~300M params); still trivial.
- **Prefix/hidden extraction**: unchanged; depth-ratio just remaps
  (36 compiler layers → L interpreter layers).
- **Training**: this is the real cost. The published run: 3 epochs ×
  10M examples, batch 48, loss = frozen-interpreter NLL of the target
  (plain SFT — no RL), fully-unfrozen 4B compiler at lr 2e-5, mapper
  fp32. ~72 h on 3 GPUs (H200-class) *for the 0.6B interpreter*. Each
  training step needs forward+backward through compiler (4B) **and**
  interpreter-with-LoRA-hooks (0.6B). Swapping in a 1.7B interpreter
  roughly grows the interpreter share ~3× → ballpark 1.5–2× total
  step cost; a 4B interpreter ~2.5–3×. So: **days, not weeks, on an
  8-GPU node — but you need the training data.**
- **Data**: the paper trains on FuzzyBench-10M. The research snapshot
  references `wtzhang-nlp/fuzzy_bench` on HF (and the pseudo-program
  cache is regenerable with `C_p` + the "examples" prompt via vLLM).
  Whether the full 10M-example set is public determines whether this
  axis is reproducible outside the authors' group. The recipe itself is
  fully specified (Appendix G) and the authors already ported it across
  interpreters once (`paw-4b-gpt2` targets a 124M GPT-2), which is
  strong evidence the same recipe transfers.

### 3c. Bigger *trained compiler* — requires retraining, likely diminishing returns

`C_L` could be initialized from a larger base (e.g. Qwen3-8B). Mapper
trunk input just tracks the compiler's hidden size. Costs scale with
compiler params (dominant share of training FLOPs); the paper offers no
evidence bigger `C_L` is the bottleneck — its ablations found the
*simplest* mapper beat richer ones, suggesting the encoding side is not
starved. If you have a training budget, 3b (bigger interpreter) is where
capability visibly grows, since the interpreter is what executes at
inference.

### Local-hardware ceiling for *compile* (this machine)

Compile-side scaling is gated by loading the compiler pair: a 4B-class
compiler fits an M4/24 GB in bf16 (~8 GB); an 8B compiler (~16 GB bf16)
would need fp8/int8 or a bigger box. The interpreter side is nearly
free — even a 4B interpreter runs quantized under llama.cpp on anything.

## 4. Serving (see `src/paw_server/`)

The official SDK is redirectable: `PAW_API_URL` (or `paw login --api-url`)
points `paw.compile()` / `paw.function()` at any server that speaks the
same REST protocol. `src/paw_server/` implements that protocol backed by
the local pipeline; the endpoints and wire formats it must honor were
read directly out of the installed SDK (`client.py`, `cache.py`):

| Endpoint | Used by | Contract |
|---|---|---|
| `POST /api/v1/compile` | `paw.compile` | JSON `{spec, compiler?, name?, tags?, public?, slug?, ephemeral?}` → `{program_id, status, slug?, compiler_snapshot?, timings?, ...}`; client timeout 120 s |
| `GET /api/v1/programs/resolve/{slug}` | `paw.function` | `{program_id}` |
| `GET /api/v1/programs/{id}/download` | `paw.function` | `.paw` bundle = **ZIP** of the program dir; `202 + Retry-After` while assets are still generating (client polls ≤ 60 s) |
| `GET /api/v1/programs/{id}` | SDK CLI | program meta JSON |
| `GET /api/v1/models/runtimes/{runtime_id}` | runtime hydration | runtime manifest (which GGUF interpreter to fetch) |
| `GET /api/v1/models/compilers` | SDK CLI | `{compilers: [...]}` |

The downloaded bundle must contain at minimum `adapter.gguf` +
`prompt_template.txt` + `meta.json` (the SDK's cache check requires the
first two; its llama.cpp runtime consumes exactly these, resolving the
base-model GGUF from `meta.json`'s `interpreter` field —
`Qwen/Qwen3-0.6B` maps to the published `Qwen3-0.6B-GGUF-Q6_K`, `gpt2` to
`GPT2-GGUF-Q8_0`). Program ids are per-compiler hashes, so the same spec
compiled with both compilers yields two distinct programs.
