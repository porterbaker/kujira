"""Call budget: compare planned calls with ``/info`` remaining calls before any --execute run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BudgetCheck:
    planned: int
    remaining: int | None  # None when /info doesn't report a limit
    buffer: int

    @property
    def ok(self) -> bool:
        return self.remaining is None or self.planned <= self.remaining - self.buffer

    def message(self) -> str:
        if self.remaining is None:
            return f"planned {self.planned} calls; /info reported no remaining-call limit"
        verdict = "OK" if self.ok else "ABORT"
        return f"{verdict}: planned {self.planned} calls vs {self.remaining} remaining - {self.buffer} buffer"


def check_budget(info: dict[str, Any], planned: int, buffer: int) -> BudgetCheck:
    remaining = info.get("remainingCalls")
    return BudgetCheck(planned, None if remaining is None else int(remaining), buffer)


def describe_info(info: dict[str, Any]) -> str:
    """One line of account status. /info carries no secrets."""
    features = ", ".join(k for k, v in (info.get("features") or {}).items() if v) or "none"
    return (
        f"tier={info.get('tierName')} (level {info.get('patronLevel')}), "
        f"remaining={info.get('remainingCalls')}/{info.get('monthlyLimit')}, "
        f"used={info.get('usedCalls')}, resets={info.get('resetAt')}, features: {features}"
    )
