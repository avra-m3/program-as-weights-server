"""Program store: on-disk artifacts + a small thread-safe registry.

Layout (under the server data dir):
    programs/<program_id>/   compile output (paw_server.compile.compile_spec + bundle)
    registry.json            {programs: {id: {...}}, slugs: {slug: id}}
"""

import hashlib
import json
import threading
import zipfile
from pathlib import Path

# Files shipped to clients inside the .paw bundle (a ZIP; the SDK cache
# requires adapter.gguf + prompt_template.txt, meta.json drives runtime
# resolution). The 80 MB PEFT safetensors stays server-side only.
BUNDLE_FILES = [
    "adapter.gguf",
    "prompt_template.txt",
    "meta.json",
    "pseudo_program.txt",
    "adapter_config.json",
]


def program_id_for(spec: str, compiler: str) -> str:
    """Deterministic 20-hex program id (idempotent recompiles)."""
    return hashlib.sha256(f"{compiler}\x00{spec}".encode()).hexdigest()[:20]


class ProgramStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.programs_dir = self.data_dir / "programs"
        self.programs_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.data_dir / "registry.json"
        self._lock = threading.Lock()
        self._registry = self._load()

    def _load(self) -> dict:
        if self._registry_path.exists():
            reg = json.loads(self._registry_path.read_text())
        else:
            reg = {"programs": {}, "slugs": {}}
        # A server restart orphans any in-flight compile.
        for entry in reg["programs"].values():
            if entry.get("status") == "compiling":
                entry["status"] = "failed"
                entry["error"] = "server restarted during compile"
        return reg

    def _save(self) -> None:
        self._registry_path.write_text(json.dumps(self._registry, indent=2))

    def program_dir(self, program_id: str) -> Path:
        return self.programs_dir / program_id

    def get(self, program_id: str) -> dict | None:
        with self._lock:
            entry = self._registry["programs"].get(program_id)
            return dict(entry) if entry else None

    def upsert(self, program_id: str, **fields) -> dict:
        with self._lock:
            entry = self._registry["programs"].setdefault(
                program_id, {"program_id": program_id}
            )
            entry.update(fields)
            self._save()
            return dict(entry)

    def bind_slug(self, slug: str, program_id: str) -> None:
        with self._lock:
            self._registry["slugs"][slug] = program_id
            self._save()

    def resolve_slug(self, slug: str) -> str | None:
        with self._lock:
            return self._registry["slugs"].get(slug)

    def list_programs(self) -> list[dict]:
        with self._lock:
            return [dict(e) for e in self._registry["programs"].values()]

    def bundle_path(self, program_id: str) -> Path:
        """Build (once) and return the .paw bundle: a ZIP of the artifacts."""
        pdir = self.program_dir(program_id)
        bundle = pdir / "bundle.paw"
        if not bundle.exists():
            tmp = bundle.with_suffix(".paw.tmp")
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                for name in BUNDLE_FILES:
                    path = pdir / name
                    if path.exists():
                        zf.write(path, arcname=name)
            tmp.replace(bundle)
        return bundle
