import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "tasks.db")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-3.5-flash")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    # Глубина «размышления» модели: minimal, low, medium, high (пусто — по умолчанию модели)
    THINKING_LEVEL: str = os.getenv("THINKING_LEVEL", "low")
    # Приятный женский голос (можно заменить на ru-RU-DmitryNeural)
    TTS_VOICE: str = os.getenv("TTS_VOICE", "ru-RU-SvetlanaNeural")
    # Запасной часовой пояс, если браузер его не передал
    DEFAULT_TIMEZONE: str = os.getenv("DEFAULT_TIMEZONE", "UTC")


settings = Settings()
