# paw-re-server

A scaffold for defining your own [ProgramAsWeights](https://github.com/programasweights/programasweights-python)
(PAW) programs and running them against test data.

PAW compiles natural-language specs into tiny neural functions that run
locally (no API key needed at inference time, deterministic). This repo
gives you a small, task-agnostic harness for that workflow: write a spec,
write test cases, compile, evaluate, iterate.

## Project layout

```
src/paw_re_server/
  models.py   # TestCase / EvalResult / SuiteReport dataclasses
  data.py     # loaders for spec .txt files and YAML test-case files
  runner.py   # compiles/loads a PAW function and runs it over test cases
specs/        # natural-language PAW specs, one .txt file per task
data/         # YAML test cases, one .yaml file per task
tests/        # pytest unit tests for the harness itself (no network calls)
```

No task has been chosen yet, so `specs/` and `data/` only contain README
files describing the conventions/schema. See `specs/README.md` and
`data/README.md`.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs `programasweights` from its package index (already configured
in `pyproject.toml` via `[[tool.uv.index]]`), plus dev tooling.

Optional: sign in to PAW for higher rate limits (copy `.env.example` to
`.env` and fill in `PAW_API_KEY`, or run `uv run paw login`).

## Running the test harness

Once a spec and test-data file exist:

```bash
uv run python -m paw_re_server.runner \
    --spec specs/<task>.txt \
    --data data/<task>.yaml
```

This compiles the spec (prints the resulting `program.id`), runs every test
case, and prints a pass/fail report with overall accuracy.

To avoid recompiling (and burning rate limit) on every run, reuse the
printed program id/slug:

```bash
uv run python -m paw_re_server.runner \
    --program-id <id-or-slug> \
    --data data/<task>.yaml
```

Exit code is `0` if all test cases pass, `1` otherwise — usable in CI.

## Local (self-hosted) compile — `paw_local`

`src/paw_local/` reimplements the PAW *compile* step so specs can be
compiled entirely on this machine, without the hosted API. It loads the
published compiler weights (`programasweights/paw-4b-qwen3-0.6b` on
Hugging Face: an 8 GB fine-tuned Qwen3-4B + the 652 MB `lora_mapper.pt`)
and reproduces the paper's Text-to-LoRA forward pass (arXiv:2607.02512
§3.2). See `plan.md` for the full research trail; the implementation
conventions (layer alignment, coefficient layout, prompts) were verified
against the authors' research snapshot and a production program artifact.

Compile is two stages, each loading a 4B model (sequentially, ~8 GB each):
the off-the-shelf `Qwen/Qwen3-4B-Instruct-2507` writes the pseudo-program,
then the trained PAW compiler encodes it into LoRA weights via the mapper.
First use downloads ~18 GB of weights to the HF cache. Note: compile wants
most of a 24 GB machine to itself — close memory-heavy apps first, or
generation will crawl due to swapping.

```bash
# Compile a spec into a program directory:
uv run paw-local compile "Classify text sentiment as positive or negative." \
    -o data/programs/sentiment

# Run the compiled program (loads only the 0.6B interpreter + adapter):
uv run paw-local run data/programs/sentiment "I love this"
```

The output directory matches the official artifact layout
(`adapter_config.json`, `adapter_model.safetensors`, `prompt_template.txt`,
`pseudo_program.txt`, `meta.json`), so `paw-local run` also works on
programs downloaded from `programasweights/paw-programs`, and the adapter
is a standard PEFT directory (convertible to `.paw` via `paw.from_peft`,
or to GGUF with `--gguf` for llama.cpp).

## Self-hosted compile server — `paw_server`

`src/paw_server/` is a FastAPI implementation of the programasweights.com
REST protocol, backed by the local pipeline — so the **unmodified official
SDK** works against it. Mechanism and wire-contract details:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

```bash
# Terminal 1: start the server (programs stored under data/server/)
uv run paw-server --port 8100

# Terminal 2: use the official SDK, pointed at it
PAW_API_URL=http://127.0.0.1:8100 uv run python -c "
import programasweights as paw
program = paw.compile('Classify text sentiment as positive or negative.')
fn = paw.function(program.id)
print(fn('I love this'))
"
```

Endpoints implemented: `POST /api/v1/compile` (blocks up to 100 s, then
the SDK's download polling takes over via `202 Accepted`),
`GET /api/v1/programs/{id}/download` (`.paw` ZIP bundle with a Q4_0
`adapter.gguf`), slug resolution, program meta/listing, and the
runtime/compiler manifests. Compiles run one at a time (each needs ~8 GB
peak); identical specs dedupe to the same program id.

## Dev tooling

```bash
uv run ruff check .      # lint
uv run black .           # format
uv run pytest            # run the harness's own unit tests
```

`ruff` handles linting + import sorting; `black` owns formatting (ruff's
formatter is disabled in favor of black, per project convention).
