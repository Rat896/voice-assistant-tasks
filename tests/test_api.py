import pytest
from fastapi.testclient import TestClient

from app.agent import agent
from app.database import TaskRepository
from app.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(agent, "client", None)
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_items_empty_then_filled(client):
    assert client.get("/api/items").json() == {"items": []}
    TaskRepository.create_item("task", "Тест", "2026-10-30 09:00")
    assert len(client.get("/api/items").json()["items"]) == 1


def test_patch_marks_completed(client):
    item = TaskRepository.create_item("task", "Тест")
    res = client.patch(f"/api/items/{item['id']}", json={"status": "completed"})
    assert res.status_code == 200
    assert res.json()["item"]["status"] == "completed"


def test_patch_rejects_bad_status(client):
    item = TaskRepository.create_item("task", "Тест")
    res = client.patch(f"/api/items/{item['id']}", json={"status": "weird"})
    assert res.status_code == 422


def test_patch_missing_is_404(client):
    assert client.patch("/api/items/999", json={"status": "completed"}).status_code == 404


def test_delete_and_missing(client):
    item = TaskRepository.create_item("task", "Тест")
    res = client.delete(f"/api/items/{item['id']}")
    assert res.status_code == 200
    assert res.json()["items"] == []
    assert client.delete(f"/api/items/{item['id']}").status_code == 404


def test_chat_without_key_returns_reply(client):
    res =client.post("/api/chat", json={"message": "привет"})
    assert res.status_code == 200
    body = res.json()
    assert body["reply"] and "items" in body


def test_tts_returns_audio(client, monkeypatch):
    async def fake_tts(text):
        return b"mp3-bytes"

    monkeypatch.setattr("app.main.synthesize_speech", fake_tts)
    res = client.post("/api/tts", json={"text": "Готово"})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/mpeg"
    assert res.content == b"mp3-bytes"


def test_tts_failure_is_502(client, monkeypatch):
    async def broken_tts(text):
        raise RuntimeError("edge-tts недоступен")

    monkeypatch.setattr("app.main.synthesize_speech", broken_tts)
    assert client.post("/api/tts", json={"text": "Готово"}).status_code == 502
