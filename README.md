# program-as-weights-server

A self-hosted [FastAPI](https://fastapi.tiangolo.com/) server that implements
the [programasweights.com](https://programasweights.com) REST compile
protocol, backed by a fully-local Program-as-Weights (PAW) compile pipeline —
no hosted API is needed to compile programs. The **unmodified official PAW
SDK** ([`programasweights`](https://github.com/programasweights/programasweights-python))
works against it unchanged, just by pointing `PAW_API_URL` at this server.

For the mechanism (what compile actually does, and the wire contract this
server implements) see [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

> [!IMPORTANT]
> **This is an independent, unofficial reimplementation.** It is not
> affiliated with, endorsed by, or supported by the authors of the
> Program-as-Weights paper or the operators of programasweights.com. It was
> built by reverse-engineering the published paper, the authors' research
> code, the released model weights, and the official SDK's wire protocol (all
> credited below). It downloads the authors' publicly *available* weights from
> Hugging Face at runtime — it does not bundle or redistribute them, and note
> those weights currently ship with no explicit license (see
> [License](#license)). Any bugs or inaccuracies are this project's own, not
> the original authors'.

## Credits

This project is a serving/reimplementation layer on top of the research
described in:

> **Program-as-Weights: A Programming Paradigm for Fuzzy Functions.**
> Wentao Zhang, Liliana Hotsko, Woojeong Kim, Pengyu Nie, Stuart Shieber,
> Yuntian Deng. arXiv:2607.02512, 2026.
> <https://arxiv.org/abs/2607.02512>

All of the core ideas — the pseudo-program/LoRA-adapter program
representation, the compiler→mapper→interpreter pipeline, the FuzzyBench
dataset, and the released models — are the work of those authors. Please cite
their paper (see [Citation](#citation)) if you use this project or build on
it. The paper is released under
[CC BY 4.0](http://creativecommons.org/licenses/by/4.0/).

The prompt templates in `src/paw_server/compile/prompts.py` are reproduced
verbatim from the paper's appendix and the authors' research code.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Meaningful free RAM to compile: two 4B models are loaded sequentially, so
  peak usage is ~8 GB, but the first compile also downloads ~18 GB of
  weights to the Hugging Face cache. See
  [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for the full cost breakdown.
  Inference (running an already-compiled program) is cheap — only the 0.6B
  interpreter + adapter is loaded.

- Tested on CUDA (Linux) and MPS (macOS). The pipeline autodetects the
  available device (`torch.cuda` → `torch.mps` → `cpu`).

## Setup

```bash
git clone git@github.com:avra-m3/program-as-weights-server.git
cd program-as-weights-server
uv sync
```

This installs the server and its compile pipeline's dependencies (torch,
transformers, etc.) plus dev tooling. It does **not** install the official
`programasweights` SDK — that package pulls in `llama-cpp-python`, which has
no prebuilt wheel for Linux/arm64, and isn't needed to run the server itself.
Install it separately only if you want to run the SDK example below:

```bash
uv pip install programasweights
```

Optional: copy `.env.example` to `.env` to tweak the *SDK's* local behavior
if you've installed it separately — `PAW_GPU_LAYERS` (force CPU-only if
GPU/Metal/CUDA causes issues) and `PAW_CACHE_DIR` (where the SDK caches
downloaded models/programs). Note that `.env.example`'s `PAW_API_KEY` is for
signing in to the *hosted* programasweights.com service and has no effect on
this self-hosted server — there's no login or rate limit here.

## Run

```bash
uv run paw-server --port 8100
```

Defaults: `--host 127.0.0.1`, `--port 8100`, `--data-dir data/server`
(compiled programs and the registry are stored under `--data-dir`).

## Docker

```bash
docker build -t paw-server .
docker run -p 8100:8100 \
    --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    paw-server
```

The image exposes port `8100` and mounting the Hugging Face cache lets
downloaded model weights persist across container restarts.

## Examples

### 1. Raw REST via curl

```bash
curl -X POST http://127.0.0.1:8100/api/v1/compile \
    -H "Content-Type: application/json" \
    -d '{"spec": "Classify text sentiment as positive or negative."}'
```

Two compilers are available (see `GET /api/v1/models/compilers`): the
default `paw-4b-qwen3-0.6b` (programs run on the Qwen3-0.6B interpreter)
and the compact `paw-4b-gpt2` (programs run on a 124M GPT-2 — smaller and
faster at inference, weaker on hard tasks). Pass `"compiler":
"paw-4b-gpt2"` to target it; the same spec compiled with a different
compiler yields a distinct program id and its own runtime (`gpt2-q8_0`).

Response (`CompileRequest` -> program response):

```json
{
  "program_id": "...",
  "status": "compiled",
  "slug": null,
  "compiler_snapshot": "paw-4b-qwen3-0.6b-20260407",
  "compiler_kind": "lora",
  "pseudo_program_strategy": "examples",
  "runtime_id": "qwen3-0.6b-q6_k",
  "runtime_manifest_version": 1,
  "timings": { "...": "..." },
  "error": null,
  "version": 1,
  "version_action": "created"
}
```

Then download the compiled `.paw` bundle (a ZIP containing `adapter.gguf`,
`prompt_template.txt`, `meta.json`, ...):

```bash
curl -OJ http://127.0.0.1:8100/api/v1/programs/<program_id>/download
```

### 2. The official SDK, pointed at this server

Requires installing the SDK separately (`uv pip install programasweights` —
see [Setup](#setup)):

```bash
PAW_API_URL=http://127.0.0.1:8100 uv run python -c "
import programasweights as paw
program = paw.compile('Classify text sentiment as positive or negative.')
fn = paw.function(program.id)
print(fn('I love this'))
"
```

## Endpoints

- `POST /api/v1/compile` — compile a spec (blocks up to 100 s; the SDK's
  download polling via `202 Accepted` takes over if compile is still
  running past that)
- `POST /api/v1/compile/raw` — like `/api/v1/compile`, but you supply the
  `pseudo_program` yourself instead of having the untrained Qwen3-4B pseudo
  compiler generate one. Same request/response shape plus a required
  `pseudo_program` field; skips straight to the trained-compiler encoding
  step.
- `GET /api/v1/compile/instructions` — the static "system prompt"
  normally given to the untrained pseudo compiler when it writes a
  pseudo-program (see `paw_server/compile/prompts.py`'s `COMPILER_EXAMPLES`)
- `GET /api/v1/programs/{id}/pseudo_program` — fetch the pseudo-program
  text for an already-compiled program
- `GET /api/v1/programs/{id}/download` — download the `.paw` bundle
- `GET /api/v1/programs/resolve/{slug}` — resolve a slug to a program id
- `GET /api/v1/programs/{id}` — program metadata
- `GET /api/v1/programs/{slug}/versions` — version history for a slug
- `GET /api/v1/models/runtimes/{runtime_id}` — runtime manifest (which GGUF
  interpreter to fetch)
- `GET /api/v1/models/compilers` — available compilers
- `GET /health` — liveness check

Compiles run one at a time (each needs ~8 GB peak); identical specs dedupe
to the same program id.

## Dev tooling

```bash
uv run ruff check .      # lint
uv run black .           # format
uv run pytest            # run the test suite
```

`ruff` handles linting + import sorting; `black` owns formatting (ruff's
formatter is disabled in favor of black, per project convention).

## License

This server implementation is released under the [MIT License](LICENSE),
© 2026 Avrami H.

The MIT license covers **only the code in this repository** (with the one
exception noted below). It does *not* cover the Program-as-Weights paper, the
model weights, the FuzzyBench dataset, or the official `programasweights` SDK —
those are the property of their respective authors. See [Credits](#credits).

### Third-party artifacts this server downloads at runtime

This server does **not** bundle or redistribute any model weights or datasets —
it fetches them from Hugging Face on first use, onto the running machine. Each
artifact's license applies as defined in its own repository, and complying with
those terms is the user's responsibility:

- [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) — pseudo compiler base
- [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) — interpreter base
- [`programasweights/paw-4b-qwen3-0.6b`](https://huggingface.co/programasweights/paw-4b-qwen3-0.6b) — trained compiler + mapper (default)
- [`programasweights/paw-4b-gpt2`](https://huggingface.co/programasweights/paw-4b-gpt2) — trained compact compiler + mapper (GPT-2 interpreter)
- [`programasweights/paw-programs`](https://huggingface.co/programasweights/paw-programs) — published program artifacts
- [`programasweights/Qwen3-0.6B-GGUF-Q6_K`](https://huggingface.co/programasweights/Qwen3-0.6B-GGUF-Q6_K) — runtime interpreter (GGUF)
- [`programasweights/GPT2-GGUF-Q8_0`](https://huggingface.co/programasweights/GPT2-GGUF-Q8_0) — compact runtime interpreter (GGUF)
- [`wtzhang-nlp/fuzzy_bench`](https://huggingface.co/datasets/wtzhang-nlp/fuzzy_bench) — FuzzyBench dataset (training/eval only)

> [!WARNING]
> Some of the `programasweights/*` artifacts do not declare an explicit license
> at their source. Absent an explicit grant, default copyright applies and reuse
> rights may be unclear. If you intend to use this server for anything beyond
> personal experimentation, check each repository and confirm the terms with the
> original authors first.

### Third-party content in this repository

The prompt templates in `src/paw_server/compile/prompts.py` are reproduced
verbatim from the Program-as-Weights paper and the authors' research code. That
text is © its original authors and licensed under
[CC BY 4.0](http://creativecommons.org/licenses/by/4.0/); it is included here
with attribution and is **not** covered by this repository's MIT license.

## Citation

If you use this project, please cite the original paper:

```bibtex
@article{zhang2026paw,
  title   = {Program-as-Weights: A Programming Paradigm for Fuzzy Functions},
  author  = {Zhang, Wentao and Hotsko, Liliana and Kim, Woojeong and
             Nie, Pengyu and Shieber, Stuart and Deng, Yuntian},
  journal = {arXiv preprint arXiv:2607.02512},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.02512}
}
```
