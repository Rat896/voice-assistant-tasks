import logging
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import agent
from app.database import TaskRepository, init_db
from app.voice import synthesize_speech, transcribe_audio

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AI Voice Task & Meeting Assistant")


@app.on_event("startup")
def on_startup():
    init_db()


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    timezone: Optional[str] = None  # например, 'Europe/Chisinau'


class TTSRequest(BaseModel):
    text: str


class ItemUpdate(BaseModel):
    status: Optional[Literal["pending", "completed"]] = None
    title: Optional[str] = None
    scheduled_at: Optional[str] = None


@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/items")
def get_all_items():
    """Возвращает все задачи и встречи для отображения на панели справа."""
    return {"items": TaskRepository.get_all()}


@app.patch("/api/items/{item_id}")
def update_item(item_id: int, body: ItemUpdate):
    """Ручное изменение записи из интерфейса (например, отметка выполненной)."""
    updated = TaskRepository.update_item(
        item_id,
        title=body.title,
        scheduled_at=body.scheduled_at,
        status=body.status,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Запись не найдена")
    if body.title:
        agent.embed_later(item_id, body.title)
    return {"item": updated, "items": TaskRepository.get_all()}


@app.delete("/api/items/{item_id}")
def delete_item(item_id: int):
    if not TaskRepository.delete_item(item_id):
        raise HTTPException(status_code=404, detail="Запись не найдена")
    return {"deleted": item_id, "items": TaskRepository.get_all()}


@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    """Озвучка ответа отдельным запросом, чтобы текст и задачи приходили без ожидания TTS."""
    try:
        audio_bytes = await synthesize_speech(req.text)
    except Exception:
        logger.warning("Не удалось синтезировать речь", exc_info=True)
        raise HTTPException(status_code=502, detail="Не удалось синтезировать речь")
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Обработка текстового запроса пользователя."""
    reply_text = agent.process_message(
        req.message, session_id=req.session_id, tz=req.timezone
    )
    return {
        "user_text": req.message,
        "reply": reply_text,
        "items": TaskRepository.get_all(),
    }


@app.post("/api/voice")
async def voice_endpoint(
    audio: UploadFile = File(...),
    session_id: str = Form("default"),
    timezone: Optional[str] = Form(None),
):
    """Полный голосовой цикл: Аудио -> STT -> LLM Agent + БД -> Ответ + TTS."""
    audio_bytes = await audio.read()
    mime_type = audio.content_type or "audio/webm"

    try:
        user_text = transcribe_audio(audio_bytes, mime_type=mime_type)
    except Exception:
        logger.exception("Ошибка распознавания речи")
        return {
            "user_text": "",
            "reply": "Не удалось распознать речь. Проверьте ключ Gemini и попробуйте снова или введите запрос текстом.",
            "items": TaskRepository.get_all(),
        }

    if not user_text:
        return {
            "user_text": "",
            "reply": "Извините, я не расслышала запрос. Попробуйте сказать еще раз.",
            "items": TaskRepository.get_all(),
        }

    reply_text = agent.process_message(user_text, session_id=session_id, tz=timezone)
    return {
        "user_text": user_text,
        "reply": reply_text,
        "items": TaskRepository.get_all(),
    }
