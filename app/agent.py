import logging
import typing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import errors, types

from app.config import settings
from app.database import TaskRepository
from app.timeutils import now_in_tz

logger = logging.getLogger(__name__)

DAYS_RU = [
    "Понедельник", "Вторник", "Среда", "Четверг",
    "Пятница", "Суббота", "Воскресенье",
]

DAYS_RU_ACC = [
    "в понедельник", "во вторник", "в среду", "в четверг",
    "в пятницу", "в субботу", "в воскресенье",
]

MONTHS_RU_GEN = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]

TYPE_ACC_RU = {"task": "задачу", "reminder": "напоминание", "meeting": "встречу"}

MAX_HISTORY = 20
MAX_TOOL_ROUNDS = 5


class AssistantAgent:
    def __init__(self):
        self.client = (
            genai.Client(api_key=settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None
        )
        self.sessions: Dict[str, List[types.Content]] = {}
        self._tz: Optional[str] = None  # часовой пояс текущего запроса
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._tools = [
            self.create_record,
            self.get_schedule,
            self.search_records,
            self.modify_record,
            self.delete_record,
        ]
        self._tool_map = {tool.__name__: tool for tool in self._tools}

    def __deepcopy__(self, memo):
        return self

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        if not self.client:
            return None
        try:
            res = self.client.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=text,
            )
            return res.embeddings[0].values
        except Exception:
            logger.warning("Не удалось получить эмбеддинг", exc_info=True)
            return None

    def _store_embedding(self, item_id: int, text: str) -> None:
        emb = self._get_embedding(text)
        if emb:
            TaskRepository.set_embedding(item_id, emb)

    def embed_later(self, item_id: int, text: str) -> None:
        """Считает эмбеддинг записи в фоне, не задерживая ответ (используется и API)."""
        if self.client:
            self._executor.submit(self._store_embedding, item_id, text)

    # --- ИНСТРУМЕНТЫ (TOOLS) ДЛЯ GEMINI ---

    def create_record(
        self,
        item_type: str,
        title: str,
        scheduled_at: Optional[str] = None,
        description: str = "",
    ) -> dict:
        """Создает задачу, напоминание или встречу в базе данных.

        Args:
            item_type: Тип записи. Строго одно из значений: 'task' (задача), 'reminder' (напоминание), 'meeting' (встреча).
            title: Краткое и понятное название задачи, напоминания или встречи (например, 'Позвонить Ивану').
            scheduled_at: Дата и время в формате 'YYYY-MM-DD HH:MM'. Если время не важно для задачи, можно указать только дату с временем 09:00 или не указывать вовсе.
            description: Дополнительный контекст, если есть.
        """
        conflicts = []
        if item_type == "meeting" and scheduled_at:
            conflicts = TaskRepository.check_conflicts(scheduled_at)

        created = TaskRepository.create_item(
            item_type=item_type,
            title=title,
            scheduled_at=scheduled_at,
            description=description,
            created_at=now_in_tz(self._tz).strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.embed_later(created["id"], f"{title} {description}")
        return {
            "status": "success",
            "created_item": created,
            "conflicts_detected": conflicts,
            "conflict_warning": "На это время уже назначена другая встреча!" if conflicts else None,
        }

    def get_schedule(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        item_type: str = "all",
    ) -> dict:
        """Получает список предстоящих или запланированных задач, напоминаний и встреч за указанный период.

        Args:
            start_date: Начало периода в формате 'YYYY-MM-DD 00:00'.
            end_date: Конец периода в формате 'YYYY-MM-DD 23:59'.
            item_type: Фильтр по типу: 'all' (все), 'task' (только задачи), 'meeting' (только встречи), 'reminder' (только напоминания).
        """
        items = TaskRepository.get_items_by_period(
            start_date=start_date,
            end_date=end_date,
            item_type=item_type,
        )
        return {"count": len(items), "items": items}

    def search_records(self, query: str) -> dict:
        """Ищет ранее созданные старые или будущие задачи, встречи и напоминания по смыслу или ключевым словам.

        Args:
            query: Поисковый запрос (например: 'презентация', 'встреча с Иваном', 'договор').
        """
        emb = self._get_embedding(query)
        results = TaskRepository.search_items(query_text=query, query_embedding=emb)
        return {"query": query, "found_count": len(results), "results": results}

    def modify_record(
        self,
        item_id: int,
        new_title: Optional[str] = None,
        new_scheduled_at: Optional[str] = None,
        new_status: Optional[str] = None,
    ) -> dict:
        """Изменяет существующую запись (переносит время, меняет текст или отмечает задачу выполненной).

        Args:
            item_id: Числовой ID записи в базе данных.
            new_title: Новое название (если нужно изменить текст).
            new_scheduled_at: Новая дата и время в формате 'YYYY-MM-DD HH:MM' (при переносе задачи/встречи).
            new_status: Новый статус: 'completed' (выполнено) или 'pending' (в работе).
        """
        updated = TaskRepository.update_item(
            item_id=item_id,
            title=new_title,
            scheduled_at=new_scheduled_at,
            status=new_status,
        )
        if not updated:
            return {"status": "error", "message": f"Запись с ID {item_id} не найдена."}
        if new_title:
            self.embed_later(item_id, new_title)
        return {"status": "success", "updated_item": updated}

    def delete_record(self, item_id: int) -> dict:
        """Удаляет задачу, встречу или напоминание по ее ID.

        Args:
            item_id: Числовой ID записи для удаления.
        """
        ok = TaskRepository.delete_item(item_id)
        return {"status": "deleted" if ok else "not_found", "item_id": item_id}

    # --- ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ---

    def _build_system_prompt(self) -> str:
        now = now_in_tz(self._tz)
        weekday = DAYS_RU[now.weekday()]
        now_str = now.strftime("%Y-%m-%d %H:%M")
        return f"""Ты — умный голосовой и текстовый ассистент для управления личными задачами, напоминаниями и встречами.
Текущие дата и время пользователя: {now_str} ({weekday}).

ПРАВИЛА РАБОТЫ:
1. Понимание естественного языка: пользователь обращается в свободной форме. Вычисляй относительные даты («завтра», «через два часа», «в пятницу», «на следующей неделе») от текущих даты и времени выше и передавай в инструменты в формате 'YYYY-MM-DD HH:MM'.
   - Если сказано 'до конца недели', ставь дату на ближайшее воскресенье 23:59.
   - Если для ЗАДАЧИ ('task') указан только день (например, 'в пятницу отправить отчет'), время можно поставить 09:00 и не переспрашивать.
2. Недостаток информации и уточняющие вопросы:
   - Если пользователь просит создать ВСТРЕЧУ ('meeting') или НАПОМИНАНИЕ ('reminder'), но НЕ указал конкретное время, НЕ создавай запись, а задай один короткий уточняющий вопрос (например: «На какое время поставить встречу?»).
3. Конфликты расписания:
   - Если при создании встречи в ответе функции вернулся непустой список `conflicts_detected`, обязательно предупреди пользователя о пересечении и предложи перенести одну из встреч.
4. Контекст диалога:
   - Если пользователь говорит 'А перенеси её на завтра в 16:00', 'Отметь её выполненной' или 'Удали эту задачу', определи по истории диалога, о какой записи идет речь. Если ID неизвестен — сначала найди запись через search_records или get_schedule.
5. Стиль ответа:
   - Твой ответ будет озвучен голосом (TTS), поэтому отвечай естественно, лаконично, дружелюбно, на русском языке, без markdown-разметки, списков со звездочками и эмодзи. Даты называй словами («завтра в 15:00»).
"""

    @staticmethod
    def _friendly_error(exc: Exception) -> str:
        text = str(exc).lower()
        if "429" in text or "resource_exhausted" in text or "quota" in text:
            return "Лимит запросов к Gemini сейчас исчерпан. Подождите минуту и попробуйте снова."
        if "api key" in text or "api_key" in text or "permission" in text or "401" in text or "403" in text:
            return "Не удалось авторизоваться в Gemini. Проверьте, что ключ GEMINI_API_KEY указан верно."
        if "timeout" in text or "timed out" in text or "connect" in text:
            return "Нет связи с сервисом Gemini. Проверьте подключение к интернету и попробуйте снова."
        return "Произошла ошибка при обработке запроса. Попробуйте переформулировать или повторить позже."

    @staticmethod
    def _trim_history(history: List[types.Content]) -> List[types.Content]:
        """Оставляет последние MAX_HISTORY записей, начиная с реплики пользователя."""
        if len(history) <= MAX_HISTORY:
            return history
        trimmed = history[-MAX_HISTORY:]
        while trimmed and (
            trimmed[0].role != "user"
            or any(part.function_response for part in trimmed[0].parts or [])
        ):
            trimmed = trimmed[1:]
        return trimmed

    def _call_tool(self, call: types.FunctionCall) -> dict:
        tool = self._tool_map.get(call.name)
        if not tool:
            return {"status": "error", "message": f"Неизвестный инструмент {call.name}."}
        args = dict(call.args or {})
        hints = typing.get_type_hints(tool)
        for name, value in args.items():
            if hints.get(name) is int and isinstance(value, float):
                args[name] = int(value)
        try:
            return tool(**args)
        except Exception as exc:
            logger.exception("Ошибка в инструменте %s", call.name)
            return {"status": "error", "message": str(exc)}

    def _spoken_when(self, scheduled_at: Optional[str]) -> str:
        if not scheduled_at:
            return "без даты"
        try:
            dt = datetime.strptime(scheduled_at, "%Y-%m-%d %H:%M")
        except ValueError:
            return f"на {scheduled_at}"
        days = (dt.date() - now_in_tz(self._tz).date()).days
        if days == 0:
            day = "сегодня"
        elif days == 1:
            day = "завтра"
        elif days == 2:
            day = "послезавтра"
        elif 2 < days < 7:
            day = DAYS_RU_ACC[dt.weekday()]
        else:
            day = f"{dt.day} {MONTHS_RU_GEN[dt.month - 1]}"
        return f"{day} в {dt:%H:%M}"

    def _quick_reply(self, results: List[Tuple[str, dict]]) -> Optional[str]:
        """Шаблонный ответ после успешного создания записей — без второго запроса к модели."""
        parts = []
        for name, result in results:
            if name != "create_record" or result.get("status") != "success":
                return None
            if result.get("conflicts_detected"):
                return None
            item = result["created_item"]
            kind = TYPE_ACC_RU.get(item["item_type"], "запись")
            parts.append(f"{kind} «{item['title']}» {self._spoken_when(item['scheduled_at'])}")
        return "Готово, добавила " + " и ".join(parts) + "." if parts else None

    def _generate(self, history: List[types.Content], config: types.GenerateContentConfig):
        try:
            return self.client.models.generate_content(
                model=settings.MODEL_NAME, contents=history, config=config
            )
        except errors.ClientError as exc:
            if not config.thinking_config or "thinking" not in str(exc).lower():
                raise
            logger.warning("Модель не поддерживает THINKING_LEVEL=%s, повторяю без него", settings.THINKING_LEVEL)
            config.thinking_config = None
            return self.client.models.generate_content(
                model=settings.MODEL_NAME, contents=history, config=config
            )

    def process_message(
        self,
        user_text: str,
        session_id: str = "default",
        tz: Optional[str] = None,
    ) -> str:
        if not self.client:
            return "Не настроен GEMINI_API_KEY в файле .env. Пожалуйста, добавьте ключ для работы ИИ-агента."

        self._tz = tz
        history = self.sessions.setdefault(session_id, [])
        start = len(history)
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_text)]))

        config = types.GenerateContentConfig(
            system_instruction=self._build_system_prompt(),
            tools=self._tools,
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=(
                types.ThinkingConfig(thinking_level=settings.THINKING_LEVEL)
                if settings.THINKING_LEVEL
                else None
            ),
        )

        reply = "Действие выполнено."
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                response = self._generate(history, config)
                if response.candidates and response.candidates[0].content:
                    history.append(response.candidates[0].content)
                calls = response.function_calls
                if not calls:
                    reply = response.text or reply
                    break

                results = [(call.name, self._call_tool(call)) for call in calls]
                history.append(types.Content(role="user", parts=[
                    types.Part.from_function_response(name=name, response=result)
                    for name, result in results
                ]))
                quick = self._quick_reply(results)
                if quick:
                    reply = quick
                    history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply)]))
                    break
            else:
                history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply)]))
        except Exception as exc:
            logger.exception("Ошибка при обращении к Gemini")
            del history[start:]  # убираем реплику, на которую не получили ответ
            return self._friendly_error(exc)

        self.sessions[session_id] = self._trim_history(history)
        return reply


agent = AssistantAgent()
