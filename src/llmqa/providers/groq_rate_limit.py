import asyncio
import hashlib
import math
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class _Reservation:
    tokens: int


class GroqRateGovernor:

    def __init__(
        self,
        tpm: int = 8000,
        rpm: int = 30,
        utilization: float = 0.85,
    ):
        if tpm < 1 or rpm < 1 or not 0 < utilization <= 1:
            raise ValueError("Invalid Groq rate-limit configuration")

        # Headroom allows for imperfect input/output estimates.
        self.token_capacity = max(1, math.floor(tpm * utilization))
        self.request_capacity = max(1, math.floor(rpm * utilization))
        self.tokens_per_second = self.token_capacity / 60.0
        self.requests_per_second = self.request_capacity / 60.0

        self._available_tokens = float(self.token_capacity)
        self._available_requests = float(self.request_capacity)
        self._last_refill = time.monotonic()
        self._blocked_until = 0.0
        self._expected_output_tokens = 512.0
        self._lock = threading.Lock()

    def expected_output_tokens(self) -> int:
        with self._lock:
            return math.ceil(self._expected_output_tokens * 1.15)

    def _try_reserve(self, tokens: int) -> tuple[_Reservation | None, float]:
        if tokens > self.token_capacity:
            raise ValueError(
                f"Judge request is estimated at {tokens} tokens, exceeding "
                f"the configured safe TPM capacity of {self.token_capacity}."
            )

        with self._lock:
            now = time.monotonic()
            elapsed = max(0.0, now - self._last_refill)
            self._available_tokens = min(
                self.token_capacity,
                self._available_tokens + elapsed * self.tokens_per_second,
            )
            self._available_requests = min(
                self.request_capacity,
                self._available_requests + elapsed * self.requests_per_second,
            )
            self._last_refill = now

            if now < self._blocked_until:
                return None, self._blocked_until - now

            missing_tokens = max(0.0, tokens - self._available_tokens)
            missing_requests = max(0.0, 1 - self._available_requests)

            if missing_tokens or missing_requests:
                delay = max(
                    missing_tokens / self.tokens_per_second,
                    missing_requests / self.requests_per_second,
                )
                return None, max(0.05, delay)

            self._available_tokens -= tokens
            self._available_requests -= 1
            return _Reservation(tokens=tokens), 0.0

    def acquire(self, tokens: int) -> _Reservation:
        while True:
            reservation, delay = self._try_reserve(tokens)
            if reservation is not None:
                return reservation
            time.sleep(delay)

    async def a_acquire(self, tokens: int) -> _Reservation:
        while True:
            reservation, delay = self._try_reserve(tokens)
            if reservation is not None:
                return reservation
            await asyncio.sleep(delay)

    def on_success(self, reservation: _Reservation, response: Any) -> None:
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        if input_tokens is None or output_tokens is None:
            return

        actual_tokens = max(0, input_tokens + output_tokens)
        with self._lock:
            # Refund overestimated tokens, or carry debt if undercounted.
            self._available_tokens = min(
                self.token_capacity,
                self._available_tokens + reservation.tokens - actual_tokens,
            )
            self._expected_output_tokens = (
                0.8 * self._expected_output_tokens + 0.2 * output_tokens
            )

    def on_rate_limit(self, reservation: _Reservation, delay: float) -> None:
        with self._lock:
            # A rejected request has not generated completion tokens.
            # Keep its RPM charge conservatively; refund reserved TPM.
            self._available_tokens = min(
                self.token_capacity, self._available_tokens + reservation.tokens
            )
            self._blocked_until = max(
                self._blocked_until, time.monotonic() + delay
            )


_GOVERNORS: dict[tuple[str, str, int, int], GroqRateGovernor] = {}
_GOVERNORS_LOCK = threading.Lock()


def get_governor(
    model: str, api_key: str, tpm: int = 8000, rpm: int = 30
) -> GroqRateGovernor:
    # Share budget without storing API keys as dictionary keys.
    fingerprint = hashlib.sha256(api_key.encode()).hexdigest()
    key = (model, fingerprint, tpm, rpm)
    with _GOVERNORS_LOCK:
        if key not in _GOVERNORS:
            _GOVERNORS[key] = GroqRateGovernor(tpm=tpm, rpm=rpm)
        return _GOVERNORS[key]


def retry_delay(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        value = headers.get("retry-after")
        if value:
            try:
                return max(0.5, float(value) + 0.3)
            except (TypeError, ValueError):
                pass

    value = getattr(exc, "retry_after", None)
    if isinstance(value, (int, float)) and value >= 0:
        return max(0.5, value + 0.3)

    match = re.search(
        r"Please try again in\s*([\d.]+)\s*(ms|seconds?|s)\b",
        str(exc), flags=re.IGNORECASE,
    )
    if match:
        amount = float(match.group(1))
        seconds = amount / 1000 if match.group(2).lower() == "ms" else amount
        return max(0.5, seconds + 0.3)

    # Provider supplied no delay: exponential backoff with jitter.
    return min(30.0, 2 ** (attempt - 1)) + random.uniform(0, 0.5)
