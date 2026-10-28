import io

import edge_tts
from google import genai
from google.genai import types

from app.config import settings


async def synthesize_speech(text: str) -> bytes:
    """Генерирует MP3-аудио из текста ответа ассистента."""
    clean_text = text.replace("*", "").replace("#", "").replace("`", "")
    communicate = edge_tts.Communicate(clean_text, settings.TTS_VOICE)
    audio_stream = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_stream.write(chunk["data"])
    return audio_stream.getvalue()


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    """Распознает голосовое сообщение пользователя с помощью мультимодальной модели Gemini."""
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY не задан")

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=settings.MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            "Распознай речь на этом аудио дословно на русском языке. "
            "Выведи ТОЛЬКО распознанный текст без комментариев. "
            "Если речи нет, верни пустую строку.",
        ],
    )
    return (response.text or "").strip()
