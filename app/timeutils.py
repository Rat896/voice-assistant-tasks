from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import settings


def get_tz(name: Optional[str] = None) -> ZoneInfo:
    """Возвращает часовой пояс по имени IANA (например, 'Europe/Chisinau').

    Если имя не передано или неверное, берется DEFAULT_TIMEZONE, затем UTC.
    """
    for candidate in (name, settings.DEFAULT_TIMEZONE):
        if candidate:
            try:
                return ZoneInfo(candidate)
            except (ZoneInfoNotFoundError, ValueError, KeyError):
                continue
    return ZoneInfo("UTC")


def now_in_tz(name: Optional[str] = None) -> datetime:
    return datetime.now(get_tz(name))
