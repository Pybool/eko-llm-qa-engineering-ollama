
import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from deepeval.models.base_model import DeepEvalBaseLLM


ProviderName = Literal[
    "ollama",
    "gemini",
    "openai",
    "litellm",
]


@dataclass(frozen=True)
class ModelConfig:
    provider: ProviderName
    model: str
    temperature: float = 0.0
    base_url: str | None = None
    api_key: str | None = None


class ModelProvider:

    DEFAULT_MODELS = {
        "ollama": "qwen2.5:7b",
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4.1-mini",
    }

    ALIASES = {
        "local": "ollama",
        "chatgpt": "openai",
        "gpt": "openai",
    }

    def __init__(self, config: ModelConfig):
        self.config = config

        if not config.model.strip():
            raise ValueError("Model name cannot be empty.")

        if not 0 <= config.temperature <= 2:
            raise ValueError(
                "Temperature must be between 0 and 2."
            )

    @classmethod
    def from_env(cls) -> "ModelProvider":
      

        load_dotenv(override=False)

        provider = os.getenv(
            "EVAL_PROVIDER", "ollama"
        ).strip().lower()

        provider = cls.ALIASES.get(
            provider, provider
        )

        if provider not in {
            "ollama",
            "gemini",
            "openai",
            "litellm",
        }:
            raise ValueError(
                f"Unsupported model provider: {provider}"
            )

        model = (
            os.getenv("EVAL_MODEL")
            or cls.DEFAULT_MODELS.get(provider, "")
        ).strip()

        temperature = float(
            os.getenv("EVAL_TEMPERATURE", "0")
        )

        config = ModelConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            base_url=os.getenv("EVAL_BASE_URL") or None,
            api_key=os.getenv("EVAL_API_KEY") or None,
        )

        return cls(config)

    def _get_api_key(
        self,
        *environment_names: str,
    ) -> str:

        if self.config.api_key:
            return self.config.api_key

        for name in environment_names:
            value = os.getenv(name)

            if value:
                return value

        raise ValueError(
            "Missing API key. Configure one of: "
            + ", ".join(environment_names)
            + " or EVAL_API_KEY."
        )

    def get_model(self) -> DeepEvalBaseLLM:

        cfg = self.config

        if cfg.provider == "ollama":
            from deepeval.models import OllamaModel

            return OllamaModel(
                model=cfg.model,
                base_url=(
                    cfg.base_url
                    or "http://localhost:11434"
                ),
                temperature=cfg.temperature,
            )

        if cfg.provider == "gemini":
            from deepeval.models import GeminiModel

            return GeminiModel(
                model=cfg.model,
                api_key=self._get_api_key(
                    "GOOGLE_API_KEY",
                    "GEMINI_API_KEY",
                ),
                temperature=cfg.temperature,
            )

        if cfg.provider == "openai":
            from deepeval.models import OpenAIModel

            return OpenAIModel(
                model=cfg.model,
                api_key=self._get_api_key(
                    "OPENAI_API_KEY"
                ),
                temperature=cfg.temperature,
                **(
                    {"base_url": cfg.base_url}
                    if cfg.base_url
                    else {}
                ),
            )

        if cfg.provider == "litellm":
            from deepeval.models import LiteLLMModel

            if "/" not in cfg.model:
                raise ValueError(
                    "LiteLLM requires a provider-prefixed "
                    "model name, such as "
                    "'anthropic/claude-sonnet-4-6'."
                )

            return LiteLLMModel(
                model=cfg.model,
                temperature=cfg.temperature,
                **(
                    {"base_url": cfg.base_url}
                    if cfg.base_url
                    else {}
                ),
                **(
                    {"api_key": cfg.api_key}
                    if cfg.api_key
                    else {}
                ),
            )

        raise ValueError(
            f"Unsupported provider: {cfg.provider}"
        )
