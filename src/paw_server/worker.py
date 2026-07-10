"""Single-flight compile worker.

Compiles run one at a time on a dedicated thread: each job loads two 4B
models sequentially (~8 GB peak), so concurrency would only cause
swapping. Duplicate submissions for an id already queued or running are
coalesced; callers wait on a per-program event.
"""

import queue
import threading
import time
import traceback

from paw_server.store import ProgramStore

COMPILER_SNAPSHOT = "paw-4b-qwen3-0.6b-20260407"
RUNTIME_ID = "qwen3-0.6b-q6_k"


class CompileWorker:
    def __init__(self, store: ProgramStore) -> None:
        self._store = store
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._events: dict[str, threading.Event] = {}
        self._specs: dict[str, str] = {}
        self._pseudo_programs: dict[str, str | None] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(
        self, program_id: str, spec: str, pseudo_program: str | None = None
    ) -> threading.Event:
        """Queue a compile (idempotent); returns the completion event.

        If ``pseudo_program`` is given, the compile pipeline skips running
        the untrained pseudo compiler and uses this text directly (backs
        POST /api/v1/compile/raw).
        """
        with self._lock:
            event = self._events.get(program_id)
            if event is not None and event.is_set():
                entry = self._store.get(program_id)
                if entry and entry.get("status") == "failed":
                    event = None  # previous attempt failed; recompile
            if event is None:
                event = threading.Event()
                self._events[program_id] = event
                self._specs[program_id] = spec
                self._pseudo_programs[program_id] = pseudo_program
                self._store.upsert(program_id, spec=spec, status="compiling")
                self._queue.put(program_id)
        return event

    def event_for(self, program_id: str) -> threading.Event | None:
        with self._lock:
            return self._events.get(program_id)

    def _run(self) -> None:
        while True:
            program_id = self._queue.get()
            if program_id is None:
                return
            spec = self._specs[program_id]
            pseudo_program = self._pseudo_programs.get(program_id)
            t0 = time.perf_counter()
            try:
                from paw_server.compile.pipeline import compile_spec

                out_dir = self._store.program_dir(program_id)
                compile_spec(
                    spec, out_dir, write_gguf=True, pseudo_program=pseudo_program
                )
                self._store.bundle_path(program_id)  # pre-build the ZIP
                self._store.upsert(
                    program_id,
                    status="compiled",
                    timings={"total_s": round(time.perf_counter() - t0, 1)},
                    error=None,
                    pseudo_program_strategy=(
                        "provided" if pseudo_program else "examples"
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - job boundary
                traceback.print_exc()
                self._store.upsert(program_id, status="failed", error=str(exc))
            finally:
                with self._lock:
                    self._events[program_id].set()
                    # Leave the event for late waiters; drop the spec.
                    self._specs.pop(program_id, None)
                    self._pseudo_programs.pop(program_id, None)
