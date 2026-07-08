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
    def fake_compile_spec(spec, out_dir, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "adapter.gguf").write_bytes(b"GGUF-fake")
        (out_dir / "prompt_template.txt").write_text(
            "x\n\n[INPUT]\n{INPUT_PLACEHOLDER}\n[END_INPUT]"
        )
        (out_dir / "meta.json").write_text(
            json.dumps({"interpreter": "Qwen/Qwen3-0.6B", "spec": spec})
        )
        (out_dir / "pseudo_program.txt").write_text("[PSEUDO_PROGRAM]...")
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
