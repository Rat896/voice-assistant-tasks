from datetime import timedelta

from google.genai import types

from app.agent import AssistantAgent
from app.database import TaskRepository
from app.timeutils import now_in_tz


def tomorrow():
    return f"{now_in_tz(None).date() + timedelta(days=1):%Y-%m-%d}"


def fresh_agent():
    a = AssistantAgent()
    a.client = None  # без сети и ключа: проверяем только инструменты
    return a


def test_create_record_and_conflict_warning():
    a = fresh_agent()
    first = a.create_record("meeting", "Встреча с командой", "2026-10-29 15:00")
    assert first["conflicts_detected"] == []
    second = a.create_record("meeting", "Созвон с клиентом", "2026-10-29 15:30")
    assert len(second["conflicts_detected"]) == 1
    assert second["conflict_warning"]


def test_task_never_reports_conflict():
    a = fresh_agent()
    a.create_record("meeting", "Встреча", "2026-10-29 15:00")
    res = a.create_record("task", "Задача", "2026-10-29 15:00")
    assert res["conflicts_detected"] == []


def test_schedule_search_modify_delete_flow():
    a = fresh_agent()
    created = a.create_record("reminder", "Позвонить Ивану", "2026-10-29 10:00")
    item_id = created["created_item"]["id"]

    schedule = a.get_schedule("2026-10-29 00:00", "2026-10-29 23:59")
    assert schedule["count"] == 1

    found = a.search_records("Ивану")
    assert found["found_count"] == 1

    moved = a.modify_record(item_id, new_scheduled_at="2026-10-30 16:00")
    assert moved["updated_item"]["scheduled_at"] == "2026-10-30 16:00"

    done = a.modify_record(item_id, new_status="completed")
    assert done["updated_item"]["status"] == "completed"

    assert a.delete_record(item_id)["status"] == "deleted"
    assert a.delete_record(item_id)["status"] == "not_found"


def test_modify_missing_item_returns_error():
    assert fresh_agent().modify_record(404, new_title="x")["status"] == "error"


def test_process_message_without_key_is_polite():
    reply = fresh_agent().process_message("привет")
    assert "GEMINI_API_KEY" in reply


def test_system_prompt_contains_user_time():
    a = fresh_agent()
    a._tz = "Europe/Chisinau"
    prompt = a._build_system_prompt()
    assert "Текущие дата и время пользователя" in prompt


def test_friendly_errors():
    f = AssistantAgent._friendly_error
    assert "Лимит" in f(Exception("429 RESOURCE_EXHAUSTED"))
    assert "GEMINI_API_KEY" in f(Exception("API key not valid"))
    assert "связи" in f(Exception("connection timed out"))
    assert "ошибка" in f(Exception("что-то странное")).lower()


def test_trim_history_starts_with_user():
    history = [
        types.Content(role="user" if i % 2 == 0 else "model", parts=[types.Part.from_text(text="x")])
        for i in range(41)
    ]
    trimmed = AssistantAgent._trim_history(history)
    assert len(trimmed) <= 20
    assert trimmed[0].role == "user"


def test_trim_history_skips_orphan_function_response():
    text = lambda role: types.Content(role=role, parts=[types.Part.from_text(text="x")])
    tool_answer = types.Content(
        role="user", parts=[types.Part.from_function_response(name="get_schedule", response={})]
    )
    history = [text("user"), text("model"), tool_answer, text("model")] + [text("user"), text("model")] * 9
    trimmed = AssistantAgent._trim_history(history)
    assert len(trimmed) == 18
    assert trimmed[0].role == "user"
    assert not trimmed[0].parts[0].function_response


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        return self.responses.pop(0)

    def embed_content(self, **kwargs):
        return types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[0.1, 0.2])])


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


def model_says(*parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


def call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


def test_create_task_needs_single_model_call():
    a = AssistantAgent()
    a.client = FakeClient([model_says(call(
        "create_record", item_type="task", title="Позвонить Ивану", scheduled_at=f"{tomorrow()} 10:00",
    ))])
    reply = a.process_message("Создай задачу позвонить Ивану завтра в 10")
    a._executor.shutdown(wait=True)

    assert a.client.models.calls == 1
    assert reply == "Готово, добавила задачу «Позвонить Ивану» завтра в 10:00."
    assert [c.role for c in a.sessions["default"]] == ["user", "model", "user", "model"]
    assert TaskRepository.search_items("Позвонить", [0.1, 0.2])[0]["relevance_score"] == 1.0


def test_meeting_conflict_goes_back_to_model():
    a = AssistantAgent()
    a.client = None
    a.create_record("meeting", "Встреча", f"{tomorrow()} 15:00")
    a.client = FakeClient([
        model_says(call("create_record", item_type="meeting", title="Созвон", scheduled_at=f"{tomorrow()} 15:00")),
        model_says(types.Part.from_text(text="Внимание, на это время уже есть встреча.")),
    ])
    reply = a.process_message("Поставь созвон завтра в 15")
    a._executor.shutdown(wait=True)

    assert a.client.models.calls == 2
    assert "уже есть встреча" in reply


def test_tool_int_args_from_float():
    a = fresh_agent()
    item_id = a.create_record("task", "Задача")["created_item"]["id"]
    result = a._call_tool(types.FunctionCall(name="delete_record", args={"item_id": float(item_id)}))
    assert result == {"status": "deleted", "item_id": item_id}


def test_spoken_when():
    a = fresh_agent()
    today = now_in_tz(None).date()
    assert a._spoken_when(None) == "без даты"
    assert a._spoken_when(f"{today:%Y-%m-%d} 09:00") == "сегодня в 09:00"
    assert a._spoken_when(f"{tomorrow()} 18:30") == "завтра в 18:30"
    far = today + timedelta(days=30)
    assert a._spoken_when(f"{far:%Y-%m-%d} 12:00").startswith(f"{far.day} ")


def test_config_with_tools_survives_deepcopy():
    from google import genai
    from google.genai import types

    a = AssistantAgent()
    a.client = genai.Client(api_key="dummy")
    config = types.GenerateContentConfig(tools=[a.create_record, a.delete_record])
    copied = config.model_copy(deep=True)
    assert copied.tools[0].__self__ is a


def test_unsupported_thinking_level_retries_without_it():
    from google.genai import errors

    class PickyModels(FakeModels):
        def generate_content(self, **kwargs):
            if kwargs["config"].thinking_config:
                raise errors.ClientError(400, {"error": {
                    "code": 400, "status": "INVALID_ARGUMENT",
                    "message": "Thinking level MINIMAL is not supported for this model.",
                }})
            return super().generate_content(**kwargs)

    a = AssistantAgent()
    a.client = FakeClient([])
    a.client.models = PickyModels([model_says(types.Part.from_text(text="Привет!"))])
    assert a.process_message("привет") == "Привет!"
