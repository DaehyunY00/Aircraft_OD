from __future__ import annotations

import time
from dataclasses import dataclass, field


def format_duration(seconds: float | int | None) -> str:
    """Format seconds as a short Korean duration string."""
    if seconds is None:
        return "계산 중"
    seconds = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}시간 {minutes:02d}분 {secs:02d}초"
    if minutes:
        return f"{minutes}분 {secs:02d}초"
    return f"{secs}초"


@dataclass
class ProgressTimer:
    """Track elapsed and estimated remaining time for repeated experiment work."""

    total: int
    start_time: float = field(default_factory=time.time)
    completed: int = 0

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def average_seconds(self) -> float | None:
        if self.completed <= 0:
            return None
        return self.elapsed() / self.completed

    def remaining_seconds(self) -> float | None:
        avg = self.average_seconds()
        if avg is None:
            return None
        return max(0, self.total - self.completed) * avg

    def update(self, step: int = 1) -> None:
        self.completed = min(self.total, self.completed + step)

    def status(self) -> str:
        return (
            f"{self.completed}/{self.total} 완료 | "
            f"경과 {format_duration(self.elapsed())} | "
            f"예상 남은 시간 {format_duration(self.remaining_seconds())}"
        )

    def next_status(self) -> str:
        next_index = min(self.completed + 1, self.total)
        return (
            f"{next_index}/{self.total} 시작 | "
            f"경과 {format_duration(self.elapsed())} | "
            f"예상 남은 시간 {format_duration(self.remaining_seconds())}"
        )
