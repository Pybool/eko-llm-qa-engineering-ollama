
import requests
from typing import Any


class AuthHelper:
    def __init__(
        self,
        base_url: str = "http://localhost:4000",
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.user: dict[str, Any] | None = None

    def login(self, email: str, password: str) -> dict[str, Any]:
        """Authenticate and preserve the session cookie."""

        response = self.session.post(
            url=f"{self.base_url}/api/auth/login",
            json={
                "email": email,
                "password": password,
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict) or not isinstance(
            data.get("user"), dict
        ):
            raise ValueError("Invalid authentication response")

        self.user = data["user"]

        return self.user

    def request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> requests.Response:
        """Make an authenticated API request."""

        if self.user is None:
            raise RuntimeError("Authentication required. Call login() first.")

        return self.session.request(
            method=method,
            url=f"{self.base_url}/api/{endpoint.lstrip('/')}",
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )

    def close(self) -> None:
        self.session.close()
