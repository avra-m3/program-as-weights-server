# paw-server

A self-hosted [FastAPI](https://fastapi.tiangolo.com/) server that implements
the [programasweights.com](https://programasweights.com) REST compile
protocol, backed by a fully-local PAW compile pipeline — no hosted API is
needed to compile programs. The **unmodified official PAW SDK**
([`programasweights`](https://github.com/programasweights/programasweights-python))
works against it unchanged, just by pointing `PAW_API_URL` at this server.

For the mechanism (what compile actually does, and the wire contract this
server implements) see [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Meaningful free RAM to compile: two 4B models are loaded sequentially, so
  peak usage is ~8 GB, but the first compile also downloads ~18 GB of
  weights to the Hugging Face cache. See
  [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for the full cost breakdown.
  Inference (running an already-compiled program) is cheap — only the 0.6B
  interpreter + adapter is loaded.

## Setup

```bash
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
- `GET /api/v1/programs/{id}/download` — download the `.paw` bundle
- `GET /api/v1/programs/resolve/{slug}` — resolve a slug to a program id
- `GET /api/v1/programs/{id}` — program metadata
- `GET /api/v1/programs` — list programs (paginated)
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
