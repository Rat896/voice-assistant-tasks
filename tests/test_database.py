from app.database import TaskRepository, cosine_similarity


def make(item_type="task", title="Тест", when=None, emb=None):
    return TaskRepository.create_item(item_type, title, scheduled_at=when, embedding=emb)


def test_create_and_get_all():
    item = make("task", "Отправить отчет", "2026-10-30 09:00")
    assert item["id"] == 1
    assert item["status"] == "pending"
    assert len(TaskRepository.get_all()) == 1


def test_conflict_detected_for_same_hour_meetings():
    make("meeting", "Встреча с командой", "2026-10-28 15:00")
    conflicts = TaskRepository.check_conflicts("2026-10-28 15:30")
    assert len(conflicts) == 1
    assert conflicts[0]["title"] == "Встреча с командой"


def test_no_conflict_for_other_hour_or_non_meeting():
    make("meeting", "Встреча", "2026-10-28 15:00")
    make("task", "Задача", "2026-10-28 16:00")
    assert TaskRepository.check_conflicts("2026-10-28 16:00") == []
    assert TaskRepository.check_conflicts("2026-10-28 17:00") == []


def test_completed_meeting_is_not_a_conflict():
    item = make("meeting", "Встреча", "2026-10-28 15:00")
    TaskRepository.update_item(item["id"], status="completed")
    assert TaskRepository.check_conflicts("2026-10-28 15:00") == []


def test_period_filter_and_type_filter():
    make("task", "Раньше", "2026-10-26 10:00")
    make("meeting", "В периоде", "2026-10-28 10:00")
    make("task", "Позже", "2026-11-05 10:00")
    items = TaskRepository.get_items_by_period("2026-10-27 00:00", "2026-10-31 23:59")
    assert [i["title"] for i in items] == ["В периоде"]
    only_tasks = TaskRepository.get_items_by_period(item_type="task")
    assert {i["title"] for i in only_tasks} == {"Раньше", "Позже"}


def test_period_excludes_completed_when_asked():
    item = make("task", "Готово", "2026-10-28 10:00")
    TaskRepository.update_item(item["id"], status="completed")
    assert TaskRepository.get_items_by_period(include_completed=False) == []


def test_update_reschedule_and_status():
    item = make("meeting", "Созвон", "2026-10-28 10:00")
    updated = TaskRepository.update_item(item["id"], scheduled_at="2026-10-29 16:00")
    assert updated["scheduled_at"] == "2026-10-29 16:00"
    assert updated["title"] == "Созвон"
    done = TaskRepository.update_item(item["id"], status="completed")
    assert done["status"] == "completed"


def test_update_missing_returns_none():
    assert TaskRepository.update_item(999, title="x") is None


def test_delete():
    item = make()
    assert TaskRepository.delete_item(item["id"]) is True
    assert TaskRepository.delete_item(item["id"]) is False
    assert TaskRepository.get_all() == []


def test_search_lexical():
    make("task", "Подготовить презентацию для Ивана")
    make("task", "Купить молоко")
    found = TaskRepository.search_items("презентацию")
    assert len(found) == 1
    assert "презентацию" in found[0]["title"]


def test_search_semantic_uses_embeddings():
    make("task", "Договор", emb=[1.0, 0.0, 0.0])
    make("task", "Другое", emb=[0.0, 1.0, 0.0])
    found = TaskRepository.search_items("контракт", query_embedding=[0.9, 0.1, 0.0])
    assert [i["title"] for i in found] == ["Договор"]


def test_search_does_not_leak_embedding():
    make("task", "Ключевое слово", emb=[1.0, 0.0])
    found = TaskRepository.search_items("ключевое")
    assert "embedding" not in found[0]


def test_cosine_similarity_edge_cases():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
