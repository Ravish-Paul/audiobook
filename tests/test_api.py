"""End-to-end API tests.

Run with:  pytest -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Use an isolated storage folder for the tests, before the app is imported.
TEST_STORAGE = Path(__file__).resolve().parent / "_test_storage"
os.environ["STORAGE_DIR"] = str(TEST_STORAGE)

from app.main import app  # noqa: E402  (import after env is configured)


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def project_id(client: TestClient) -> str:
    response = client.post(
        "/projects",
        json={"name": "Test Audiobook", "description": "pytest project"},
    )
    assert response.status_code == 201
    return response.json()["project_id"]


# --------------------------------------------------------------------------- #
# Basic endpoints
# --------------------------------------------------------------------------- #
def test_welcome(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body
    assert body["docs_url"] == "/docs"


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["storage_ready"] is True


# --------------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------------- #
def test_create_project_generates_uuid(client: TestClient) -> None:
    response = client.post("/projects", json={"name": "Project One"})
    assert response.status_code == 201
    body = response.json()
    assert len(body["project_id"]) == 36
    assert body["project_id"].count("-") == 4  # standard UUID shape
    assert body["name"] == "Project One"
    assert body["has_script"] is False


def test_create_project_requires_name(client: TestClient) -> None:
    response = client.post("/projects", json={})
    assert response.status_code == 422


def test_get_project_404(client: TestClient) -> None:
    response = client.get("/projects/does-not-exist")
    assert response.status_code == 404


def test_get_project_details(client: TestClient, project_id: str) -> None:
    response = client.get(f"/projects/{project_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == project_id
    assert body["name"] == "Test Audiobook"
    assert body["has_script"] is False


# --------------------------------------------------------------------------- #
# Script upload — the first goal: a successful .txt upload
# --------------------------------------------------------------------------- #
def test_upload_txt_script_success(client: TestClient, project_id: str) -> None:
    content = "नमस्ते दोस्तों!\nThis is my first audiobook script.\nLine three."
    response = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", content.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["project_id"] == project_id
    assert body["filename"] == "script.txt"
    assert body["character_count"] == len(content)
    assert body["upload_status"] == "success"

    # File really exists on disk at storage/projects/{id}/script.txt
    stored = TEST_STORAGE / project_id / "script.txt"
    assert stored.is_file()
    assert stored.read_text(encoding="utf-8") == content

    # ...and the project now reports the script
    details = client.get(f"/projects/{project_id}").json()
    assert details["has_script"] is True
    assert details["script"]["character_count"] == len(content)


def test_upload_unknown_project_404(client: TestClient) -> None:
    response = client.post(
        "/projects/00000000-0000-0000-0000-000000000000/upload-script",
        files={"file": ("script.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 404


def test_upload_rejects_non_txt(client: TestClient, project_id: str) -> None:
    response = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("notes.md", b"# hello", "text/markdown")},
    )
    assert response.status_code == 400


def test_upload_rejects_invalid_utf8(client: TestClient, project_id: str) -> None:
    response = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", b"\xff\xfe\x00invalid", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_file_over_5mb(client: TestClient, project_id: str) -> None:
    oversized = b"a" * (5 * 1024 * 1024 + 1)
    response = client.post(
        f"/projects/{project_id}/upload-script",
        files={"file": ("script.txt", oversized, "text/plain")},
    )
    assert response.status_code == 413
    assert not (TEST_STORAGE / project_id / "script.txt").exists()
