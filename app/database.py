import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import settings

PUBLIC_COLUMNS = "id, item_type, title, description, scheduled_at, status, created_at"


def get_connection():
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_type TEXT NOT NULL,            -- 'task' (задача), 'meeting' (встреча), 'reminder' (напоминание)
            title TEXT NOT NULL,                -- Название или текст
            description TEXT DEFAULT '',        -- Дополнительные детали
            scheduled_at TEXT,                  -- Дата и время в формате ISO (YYYY-MM-DD HH:MM)
            status TEXT DEFAULT 'pending',      -- 'pending' (активно) или 'completed' (выполнено)
            created_at TEXT NOT NULL,           -- Дата создания
            embedding TEXT                      -- JSON-вектор для семантического поиска
        )
        """
    )
    conn.commit()
    conn.close()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class TaskRepository:
    @staticmethod
    def create_item(
        item_type: str,
        title: str,
        scheduled_at: Optional[str] = None,
        description: str = "",
        embedding: Optional[List[float]] = None,
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        conn = get_connection()
        cursor = conn.cursor()
        if created_at is None:
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emb_json = json.dumps(embedding) if embedding else None
        cursor.execute(
            """
            INSERT INTO items (item_type, title, description, scheduled_at, status, created_at, embedding)
            VALUES (?, ?, ?, ?, 'pending', ?, ?)
            """,
            (item_type, title, description, scheduled_at, created_at, emb_json),
        )
        item_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return {
            "id": item_id,
            "item_type": item_type,
            "title": title,
            "description": description,
            "scheduled_at": scheduled_at,
            "status": "pending",
            "created_at": created_at,
        }

    @staticmethod
    def check_conflicts(scheduled_at: str) -> List[Dict[str, Any]]:
        """Проверяет, есть ли уже встречи на этот же час (бонусный критерий ТЗ)."""
        if not scheduled_at or len(scheduled_at) < 13:
            return []
        hour_prefix = scheduled_at[:13]  # Например: '2026-10-28 15'
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, item_type, title, scheduled_at, status
            FROM items
            WHERE scheduled_at LIKE ? AND status = 'pending' AND item_type = 'meeting'
            """,
            (f"{hour_prefix}%",),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    @staticmethod
    def get_items_by_period(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        item_type: Optional[str] = None,
        include_completed: bool = True,
    ) -> List[Dict[str, Any]]:
        conn = get_connection()
        cursor = conn.cursor()
        query = f"SELECT {PUBLIC_COLUMNS} FROM items WHERE 1=1"
        params: list = []
        if start_date:
            query += " AND scheduled_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND scheduled_at <= ?"
            params.append(end_date)
        if item_type and item_type != "all":
            query += " AND item_type = ?"
            params.append(item_type)
        if not include_completed:
            query += " AND status = 'pending'"
        query += " ORDER BY CASE WHEN scheduled_at IS NULL THEN 1 ELSE 0 END, scheduled_at ASC"
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    @staticmethod
    def search_items(
        query_text: str, query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Гибридный поиск: точное совпадение подстроки + векторный семантический поиск."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT {PUBLIC_COLUMNS}, embedding FROM items")
        all_rows = cursor.fetchall()
        conn.close()

        results = []
        q_lower = query_text.lower()
        for row in all_rows:
            item = dict(row)
            emb_str = item.pop("embedding", None)
            score = 0.0

            # 1. Лексическое совпадение
            if q_lower in item["title"].lower() or q_lower in (item["description"] or "").lower():
                score += 0.5

            # 2. Семантическое сходство по векторам
            if query_embedding and emb_str:
                try:
                    item_emb = json.loads(emb_str)
                    sim = cosine_similarity(query_embedding, item_emb)
                    score = max(score, sim)
                except Exception:
                    pass

            if score >= 0.42:
                item["relevance_score"] = round(score, 3)
                results.append(item)

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:5]

    @staticmethod
    def update_item(
        item_id: int,
        title: Optional[str] = None,
        scheduled_at: Optional[str] = None,
        status: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM items WHERE id = ?", (item_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        new_title = title if title is not None else row["title"]
        new_sched = scheduled_at if scheduled_at is not None else row["scheduled_at"]
        new_status = status if status is not None else row["status"]
        new_emb = json.dumps(embedding) if embedding is not None else row["embedding"]

        cursor.execute(
            """
            UPDATE items
            SET title = ?, scheduled_at = ?, status = ?, embedding = ?
            WHERE id = ?
            """,
            (new_title, new_sched, new_status, new_emb, item_id),
        )
        conn.commit()
        cursor.execute(f"SELECT {PUBLIC_COLUMNS} FROM items WHERE id = ?", (item_id,))
        updated = dict(cursor.fetchone())
        conn.close()
        return updated

    @staticmethod
    def set_embedding(item_id: int, embedding: List[float]) -> None:
        conn = get_connection()
        conn.execute(
            "UPDATE items SET embedding = ? WHERE id = ?", (json.dumps(embedding), item_id)
        )
        conn.commit()
        conn.close()

    @staticmethod
    def delete_item(item_id: int) -> bool:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM items WHERE id = ?", (item_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    @staticmethod
    def get_all() -> List[Dict[str, Any]]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT {PUBLIC_COLUMNS}
            FROM items
            ORDER BY status DESC, CASE WHEN scheduled_at IS NULL THEN 1 ELSE 0 END, scheduled_at ASC
            """
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows
