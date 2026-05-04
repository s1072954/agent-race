from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class AsyncSlidingWindowLimiter:
    global_rpm: int
    per_model_rpm: int
    window_seconds: int = 60
    _global_hits: deque[float] = field(default_factory=deque)
    _model_hits: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(self, model: str) -> None:
        if self.global_rpm <= 0 and self.per_model_rpm <= 0:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                self._prune(self._global_hits, now)
                model_hits = self._model_hits[model]
                self._prune(model_hits, now)

                global_ok = self.global_rpm <= 0 or len(self._global_hits) < self.global_rpm
                model_ok = self.per_model_rpm <= 0 or len(model_hits) < self.per_model_rpm
                if global_ok and model_ok:
                    self._global_hits.append(now)
                    model_hits.append(now)
                    return

                wait_for = self._next_wait(now, self._global_hits, self.global_rpm)
                wait_for = max(wait_for, self._next_wait(now, model_hits, self.per_model_rpm))
            await asyncio.sleep(max(0.25, min(wait_for, 10.0)))

    def _prune(self, hits: deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()

    def _next_wait(self, now: float, hits: deque[float], limit: int) -> float:
        if limit <= 0 or len(hits) < limit or not hits:
            return 0.25
        return self.window_seconds - (now - hits[0]) + 0.05
