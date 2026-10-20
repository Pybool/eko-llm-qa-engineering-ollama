import logging
import math
import os

from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from litellm import acompletion, completion, token_counter
from litellm.exceptions import RateLimitError

from llmqa.providers.groq_rate_limit import get_governor, retry_delay

logger = logging.getLogger(__name__)


class GroqRateAwareModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "groq/openai/gpt-oss-120b",
        api_key: str | None = None,
        temperature: float = 0,
        base_url: str | None = None,
        tokens_per_minute: int = 8000,
        requests_per_minute: int = 30,
        max_attempts: int = 5,
        max_retry_wait_seconds: float = 90,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required")
        if not model.startswith("groq/"):
            raise ValueError("Expected LiteLLM model name prefixed with 'groq/'")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.temperature = temperature
        self.base_url = base_url
        self.max_attempts = max_attempts
        self.max_retry_wait_seconds = max_retry_wait_seconds
        self.governor = get_governor(
            model, self.api_key,
            tpm=tokens_per_minute, rpm=requests_per_minute,
        )
        super().__init__(model=model)

    def load_model(self):
        return None

    def get_model_name(self) -> str:
        return self.name

    def _parameters(self, prompt: str, schema: type[BaseModel] | None) -> dict:
        messages = [{"role": "user", "content": prompt}]
        params = {
            "model": self.name,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "messages": messages,
        }
        if self.base_url:
            params["api_base"] = self.base_url
        if schema is not None:
            messages.insert(0, {
                "role": "system",
                "content": "Return only valid JSON matching the requested format. "
                           "Each verdict must have a verdict and reason field. "
                           "Use null for an optional reason. No Markdown.",
            })
            params["response_format"] = {"type": "json_object"}
        return params

    def _estimate(self, messages: list[dict]) -> int:
        try:
            input_tokens = token_counter(model=self.name, messages=messages)
        except Exception:
            input_tokens = math.ceil(sum(len(m["content"]) for m in messages) / 3)
        return math.ceil(input_tokens * 1.1) + self.governor.expected_output_tokens()

    @staticmethod
    def _parse(response, schema: type[BaseModel] | None):
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq judge returned an empty response")
        return schema.model_validate_json(content) if schema else content

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs):
        params = self._parameters(str(prompt), schema)
        tokens = self._estimate(params["messages"])

        for attempt in range(1, self.max_attempts + 1):
            reservation = self.governor.acquire(tokens)
            try:
                response = completion(**params)
                self.governor.on_success(reservation, response)
                return self._parse(response, schema)
            except RateLimitError as exc:
                delay = retry_delay(exc, attempt)
                self.governor.on_rate_limit(reservation, delay)
                if attempt == self.max_attempts or delay > self.max_retry_wait_seconds:
                    raise
                logger.warning(
                    "Groq judge throttled (attempt %d/%d); retry in %.2fs",
                    attempt, self.max_attempts, delay,
                )

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs):
        params = self._parameters(str(prompt), schema)
        tokens = self._estimate(params["messages"])

        for attempt in range(1, self.max_attempts + 1):
            reservation = await self.governor.a_acquire(tokens)
            try:
                response = await acompletion(**params)
                self.governor.on_success(reservation, response)
                return self._parse(response, schema)
            except RateLimitError as exc:
                delay = retry_delay(exc, attempt)
                self.governor.on_rate_limit(reservation, delay)
                if attempt == self.max_attempts or delay > self.max_retry_wait_seconds:
                    raise
                logger.warning(
                    "Groq judge throttled (attempt %d/%d); retry in %.2fs",
                    attempt, self.max_attempts, delay,
                )
