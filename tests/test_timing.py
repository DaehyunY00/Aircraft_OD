from src.utils.timing import ProgressTimer, format_duration


def test_format_duration_korean_units() -> None:
    assert format_duration(0) == "0초"
    assert format_duration(65) == "1분 05초"
    assert format_duration(3661) == "1시간 01분 01초"


def test_progress_timer_status() -> None:
    timer = ProgressTimer(total=4)
    assert "예상 남은 시간 계산 중" in timer.status()
    timer.update()
    status = timer.status()
    assert "1/4 완료" in status
    assert "경과" in status
