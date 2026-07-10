"""Protocol tests for the local PAW compile server (compile pipeline stubbed)."""

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from paw_server.app import create_app
from paw_server.store import program_id_for


@pytest.fixture()
def client(tmp_path, monkeypatch):
    def fake_compile_spec(spec, out_dir, pseudo_program=None, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "adapter.gguf").write_bytes(b"GGUF-fake")
        (out_dir / "prompt_template.txt").write_text(
            "x\n\n[INPUT]\n{INPUT_PLACEHOLDER}\n[END_INPUT]"
        )
        (out_dir / "meta.json").write_text(
            json.dumps({"interpreter": "Qwen/Qwen3-0.6B", "spec": spec})
        )
        (out_dir / "pseudo_program.txt").write_text(
            pseudo_program.strip() if pseudo_program else "[PSEUDO_PROGRAM]..."
        )
        (out_dir / "adapter_config.json").write_text("{}")
        return out_dir

    import paw_server.compile.pipeline

    monkeypatch.setattr(paw_server.compile.pipeline, "compile_spec", fake_compile_spec)
    return TestClient(create_app(tmp_path / "server"))


def test_compile_is_idempotent_and_returns_program_id(client):
    r1 = client.post("/api/v1/compile", json={"spec": "count words"})
    assert r1.status_code == 200
    body = r1.json()
    assert body["program_id"] == program_id_for("count words", "paw-4b-qwen3-0.6b")
    assert body["status"] == "compiled"
    assert body["compiler_kind"] == "lora"

    r2 = client.post("/api/v1/compile", json={"spec": "count words"})
    assert r2.json()["program_id"] == body["program_id"]
    assert r2.json()["version_action"] == "no_change"


def test_name_and_slug_distinguish_identical_specs(client):
    base = client.post("/api/v1/compile", json={"spec": "same spec"}).json()
    named = client.post(
        "/api/v1/compile", json={"spec": "same spec", "name": "variant-a"}
    ).json()
    slugged = client.post(
        "/api/v1/compile", json={"spec": "same spec", "slug": "me/variant-b"}
    ).json()

    # Identical spec text, differing identity => distinct program ids.
    assert base["program_id"] != named["program_id"]
    assert base["program_id"] != slugged["program_id"]
    assert named["program_id"] != slugged["program_id"]

    # Repeating the same spec+name is still idempotent.
    named2 = client.post(
        "/api/v1/compile", json={"spec": "same spec", "name": "variant-a"}
    ).json()
    assert named2["program_id"] == named["program_id"]
    assert named2["version_action"] == "no_change"


def test_unknown_compiler_is_422(client):
    r = client.post("/api/v1/compile", json={"spec": "x", "compiler": "nope"})
    assert r.status_code == 422


def test_download_bundle_is_zip_with_required_files(client):
    pid = client.post("/api/v1/compile", json={"spec": "s"}).json()["program_id"]
    r = client.get(f"/api/v1/programs/{pid}/download")
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = set(zf.namelist())
    # The SDK cache requires these two; meta.json drives runtime resolution.
    assert {"adapter.gguf", "prompt_template.txt", "meta.json"} <= names


def test_download_unknown_program_is_404_not_found(client):
    r = client.get("/api/v1/programs/00000000000000000000/download")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_slug_binding_and_resolution(client):
    pid = client.post(
        "/api/v1/compile", json={"spec": "s2", "slug": "me/my-prog"}
    ).json()["program_id"]
    r = client.get("/api/v1/programs/resolve/me/my-prog")
    assert r.status_code == 200
    assert r.json()["program_id"] == pid


def test_runtime_manifest_matches_sdk_expectations(client):
    r = client.get("/api/v1/models/runtimes/qwen3-0.6b-q6_k")
    assert r.status_code == 200
    m = r.json()
    assert m["local_sdk"]["supported"] is True
    assert m["local_sdk"]["base_model"]["file"] == "qwen3-0.6b-q6_k.gguf"


def test_program_meta_and_listing(client):
    pid = client.post("/api/v1/compile", json={"spec": "s3"}).json()["program_id"]
    meta = client.get(f"/api/v1/programs/{pid}").json()
    assert meta["interpreter"] == "Qwen/Qwen3-0.6B"
    listing = client.get("/api/v1/programs", params={"mine": "true"}).json()
    assert any(p["program_id"] == pid for p in listing["programs"])


def test_compile_instructions_returns_static_prompt(client):
    r = client.get("/api/v1/compile/instructions")
    assert r.status_code == 200
    body = r.json()
    assert "instructions" in body
    assert body["instructions"].startswith("You are PAW-Compiler.")
    assert "[END_PSEUDO_PROGRAM]" in body["instructions"]
    # Unfilled placeholder -- this is a template, not a rendered prompt.
    assert "{spec}" in body["instructions"]


def test_compile_raw_uses_supplied_pseudo_program(client):
    r = client.post(
        "/api/v1/compile/raw",
        json={"spec": "count words", "pseudo_program": "Task: count the words."},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "compiled"
    assert body["pseudo_program_strategy"] == "provided"

    pseudo = client.get(f"/api/v1/programs/{body['program_id']}/pseudo_program")
    assert pseudo.status_code == 200
    assert pseudo.json()["pseudo_program"] == "Task: count the words."


def test_compile_raw_rejects_empty_pseudo_program(client):
    r = client.post(
        "/api/v1/compile/raw", json={"spec": "count words", "pseudo_program": "   "}
    )
    assert r.status_code == 422


def test_compile_raw_gets_distinct_program_id_from_normal_compile(client):
    normal = client.post("/api/v1/compile", json={"spec": "same spec"}).json()
    raw = client.post(
        "/api/v1/compile/raw",
        json={"spec": "same spec", "pseudo_program": "Task: custom."},
    ).json()
    assert normal["program_id"] != raw["program_id"]
    assert normal["pseudo_program_strategy"] == "examples"
    assert raw["pseudo_program_strategy"] == "provided"


def test_get_pseudo_program_for_normal_compile(client):
    pid = client.post("/api/v1/compile", json={"spec": "s4"}).json()["program_id"]
    r = client.get(f"/api/v1/programs/{pid}/pseudo_program")
    assert r.status_code == 200
    body = r.json()
    assert body["program_id"] == pid
    assert body["pseudo_program"] == "[PSEUDO_PROGRAM]..."


def test_get_pseudo_program_unknown_program_is_404(client):
    r = client.get("/api/v1/programs/00000000000000000000/pseudo_program")
    assert r.status_code == 404
