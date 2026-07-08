"""FastAPI app implementing the programasweights.com REST protocol.

Endpoint contracts follow the installed SDK (programasweights.client /
.cache): see docs/HOW_IT_WORKS.md §4 for the mapping.
"""

import json
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from paw_server.store import ProgramStore, program_id_for
from paw_server.worker import COMPILER_SNAPSHOT, RUNTIME_ID, CompileWorker

DEFAULT_COMPILER = "paw-4b-qwen3-0.6b"

# How long POST /compile blocks waiting for the result. The SDK's client
# timeout is 120 s; if we're still compiling at the deadline we return
# status "compiling" and the client's download polling (202, <=60 s more)
# picks up from there.
COMPILE_WAIT_S = 100

# Mirrors the SDK's built-in manifest for this runtime
# (programasweights.cache.LEGACY_RUNTIME_MANIFESTS).
RUNTIME_MANIFESTS = {
    RUNTIME_ID: {
        "runtime_id": RUNTIME_ID,
        "manifest_version": 1,
        "display_name": "Qwen3 0.6B (Q6_K)",
        "interpreter": "Qwen/Qwen3-0.6B",
        "adapter_format": "gguf_lora",
        "local_sdk": {
            "supported": True,
            "base_model": {
                "provider": "huggingface",
                "repo": "programasweights/Qwen3-0.6B-GGUF-Q6_K",
                "file": "qwen3-0.6b-q6_k.gguf",
                "url": (
                    "https://huggingface.co/programasweights/"
                    "Qwen3-0.6B-GGUF-Q6_K/resolve/main/qwen3-0.6b-q6_k.gguf"
                ),
                "sha256": None,
            },
            "n_ctx": 2048,
        },
        "js_sdk": {
            "supported": False,
            "base_model": None,
            "prefix_cache_supported": False,
        },
    }
}


class CompileRequest(BaseModel):
    spec: str
    compiler: str | None = None
    name: str | None = None
    tags: list[str] | None = None
    public: bool = True
    slug: str | None = None
    ephemeral: bool = False


def create_app(data_dir: str | Path) -> FastAPI:
    store = ProgramStore(data_dir)
    worker = CompileWorker(store)
    app = FastAPI(title="paw-server (local PAW compile)")

    def _program_response(entry: dict) -> dict:
        return {
            "program_id": entry["program_id"],
            "status": entry.get("status", "unknown"),
            "slug": entry.get("slug"),
            "compiler_snapshot": COMPILER_SNAPSHOT,
            "compiler_kind": "lora",
            "pseudo_program_strategy": "examples",
            "runtime_id": RUNTIME_ID,
            "runtime_manifest_version": 1,
            "timings": entry.get("timings"),
            "error": entry.get("error"),
            "version": entry.get("version", 1),
            "version_action": entry.get("version_action"),
        }

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/v1/compile")
    def compile_program(req: CompileRequest) -> dict:
        compiler = req.compiler or DEFAULT_COMPILER
        if compiler not in (DEFAULT_COMPILER, COMPILER_SNAPSHOT):
            return JSONResponse(
                status_code=422,
                content={"detail": f"Unknown compiler '{compiler}'."},
            )

        program_id = program_id_for(req.spec, DEFAULT_COMPILER)
        existing = store.get(program_id)
        if existing and existing.get("status") == "compiled":
            entry = store.upsert(program_id, version_action="no_change")
        else:
            event = worker.submit(program_id, req.spec)
            event.wait(timeout=COMPILE_WAIT_S)
            entry = store.get(program_id) or {"program_id": program_id}
            entry["version_action"] = "created"

        if req.slug:
            store.bind_slug(req.slug, program_id)
            entry = store.upsert(program_id, slug=req.slug)
            entry["version_action"] = entry.get("version_action") or "created"

        return _program_response(entry)

    @app.get("/api/v1/programs/resolve/{slug:path}")
    def resolve_slug(slug: str) -> dict:
        program_id = store.resolve_slug(slug)
        if program_id is None:
            return JSONResponse(
                status_code=404, content={"detail": f"Slug '{slug}' not found"}
            )
        return {"program_id": program_id}

    @app.get("/api/v1/programs/{program_id}/download")
    def download_program(program_id: str) -> Response:
        entry = store.get(program_id)
        if entry is None:
            return JSONResponse(
                status_code=404, content={"detail": "Program not found"}
            )
        status = entry.get("status")
        if status == "compiling":
            # SDK polls on 202 + Retry-After while assets generate.
            return Response(status_code=202, headers={"Retry-After": "5"})
        if status != "compiled":
            return JSONResponse(
                status_code=500,
                content={"detail": f"Compile failed: {entry.get('error')}"},
            )
        bundle = store.bundle_path(program_id)
        return FileResponse(
            bundle,
            media_type="application/zip",
            filename=f"{program_id}.paw",
        )

    @app.get("/api/v1/programs/{slug:path}/versions")
    def list_versions(slug: str) -> dict:
        program_id = store.resolve_slug(slug) or slug
        entry = store.get(program_id)
        versions = [_program_response(entry)] if entry else []
        return {"slug": slug, "versions": versions}

    @app.get("/api/v1/programs/{program_id}")
    def get_program(program_id: str) -> dict:
        entry = store.get(program_id)
        if entry is None:
            return JSONResponse(
                status_code=404, content={"detail": "Program not found"}
            )
        meta_path = store.program_dir(program_id) / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return {**meta, **_program_response(entry)}

    @app.get("/api/v1/programs")
    def list_programs(
        mine: bool = False, sort: str = "recent", per_page: int = 20, page: int = 1
    ) -> dict:
        programs = [_program_response(e) for e in store.list_programs()]
        start = (page - 1) * per_page
        return {
            "programs": programs[start : start + per_page],
            "total": len(programs),
            "page": page,
            "per_page": per_page,
        }

    @app.get("/api/v1/models/runtimes/{runtime_id}")
    def get_runtime(runtime_id: str) -> dict:
        manifest = RUNTIME_MANIFESTS.get(runtime_id)
        if manifest is None:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Runtime '{runtime_id}' not found"},
            )
        return manifest

    @app.get("/api/v1/models/compilers")
    def list_compilers() -> dict:
        return {
            "compilers": [
                {
                    "name": DEFAULT_COMPILER,
                    "snapshot": COMPILER_SNAPSHOT,
                    "kind": "lora",
                    "interpreter": "Qwen/Qwen3-0.6B",
                    "runtime_id": RUNTIME_ID,
                    "default": True,
                }
            ]
        }

    return app
