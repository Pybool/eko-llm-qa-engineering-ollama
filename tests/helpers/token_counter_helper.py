
from typing import Any
import requests


class TokenCounterHelper:

    def __init__(
        self,
        base_url: str = "http://localhost:4000",
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def count_tokens(
        self,
        prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Count tokens using ContextLab's public API.

        Returns token counts and the budget decision.
        """

        payload: dict[str, Any] = {
            "prompt": prompt
        }

        if history is not None:
            payload["history"] = history

        response = self.session.post(
            url=f"{self.base_url}/api/tokens/count",
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def close(self) -> None:
        self.session.close()
