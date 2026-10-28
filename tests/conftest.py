import pytest

from app import database
from app.config import settings


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Каждый тест работает с чистой временной базой данных."""
    monkeypatch.setattr(settings, "DATABASE_PATH", str(tmp_path / "test.db"))
    database.init_db()
    yield
