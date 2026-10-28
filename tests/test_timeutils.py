from app.timeutils import get_tz, now_in_tz


def test_valid_timezone():
    assert str(get_tz("Europe/Chisinau")) == "Europe/Chisinau"


def test_invalid_timezone_falls_back_to_default():
    assert str(get_tz("Not/AZone")) == "UTC"


def test_none_timezone_uses_default():
    assert str(get_tz(None)) == "UTC"


def test_now_respects_timezone():
    utc = now_in_tz("UTC")
    tokyo = now_in_tz("Asia/Tokyo")
    assert (tokyo.utcoffset() - utc.utcoffset()).total_seconds() == 9 * 3600
