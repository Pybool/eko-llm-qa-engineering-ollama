
import os
from copy import deepcopy
from typing import Optional, Type

from pydantic import BaseModel
from deepeval.models.base_model import DeepEvalBaseLLM
from litellm import completion, acompletion


class GroqStrictLiteLLMModel(DeepEvalBaseLLM):
    """
    Groq-backed DeepEval judge with strict structured output support.

    Uses LiteLLM for API requests and Pydantic for response validation.
    Designed for text-based DeepEval metrics.
    """

    def __init__(
        self,
        model: str = "groq/openai/gpt-oss-120b",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.temperature = temperature
        self.base_url = base_url

        if not self.api_key:
            raise ValueError(
                "Missing Groq API key. Set GROQ_API_KEY "
                "or provide api_key."
            )

        if not model.startswith("groq/"):
            raise ValueError(
                "Expected a Groq model prefixed with 'groq/'."
            )

        if not 0 <= temperature <= 2:
            raise ValueError(
                "Temperature must be between 0 and 2."
            )

        super().__init__(model=model)

    def load_model(self):
        """
        LiteLLM manages the underlying API client.
        No local model needs to be loaded.
        """
        return None

    def get_model_name(self) -> str:
        return self.name

    @staticmethod
    def _normalize_schema(node):
        """
        Convert Pydantic JSON Schema into a schema compatible
        with Groq strict structured outputs.

        Strict mode requires:
        - All object properties to be required.
        - additionalProperties to be false.
        - Optional fields represented using nullable types.
        """

        if isinstance(node, dict):
            node.pop("default", None)

            if node.get("type") == "object":
                properties = node.get("properties", {})

                node["required"] = list(properties.keys())
                node["additionalProperties"] = False

            for value in node.values():
                GroqStrictLiteLLMModel._normalize_schema(value)

        elif isinstance(node, list):
            for item in node:
                GroqStrictLiteLLMModel._normalize_schema(item)

        return node

    @classmethod
    def _build_response_format(
        cls,
        schema: Type[BaseModel],
    ) -> dict:
        """
        Build Groq strict JSON Schema response format.
        """

        json_schema = deepcopy(schema.model_json_schema())

        json_schema = cls._normalize_schema(json_schema)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": json_schema,
            },
        }

    def _build_request_params(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> dict:
        """
        Build LiteLLM request parameters.
        """

        params = {
            "model": self.name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.temperature,
            "api_key": self.api_key,
        }

        if self.base_url:
            params["api_base"] = self.base_url

        if schema is not None:
            params["response_format"] = {"type": "json_object"}

        return params

    @staticmethod
    def _parse_response(
        response,
        schema: Optional[Type[BaseModel]] = None,
    ):
        """
        Validate structured output using Pydantic.
        Return plain text when no schema is requested.
        """

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                "Groq returned an empty judge response."
            )

        if schema is None:
            return content

        return schema.model_validate_json(content)

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs,
    ):
        """
        Synchronous generation for DeepEval metrics.
        """

        params = self._build_request_params(
            prompt=prompt,
            schema=schema,
        )

        response = completion(**params)

        return self._parse_response(
            response=response,
            schema=schema,
        )

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        **kwargs,
    ):
        """
        Asynchronous generation for DeepEval metrics.
        """

        params = self._build_request_params(
            prompt=prompt,
            schema=schema,
        )

        response = await acompletion(**params)

        return self._parse_response(
            response=response,
            schema=schema,
        )
