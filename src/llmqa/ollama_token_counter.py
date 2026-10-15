
import json
from urllib.request import Request, urlopen

import tiktoken


class FakeTokenCounter:

    def __init__(self):
        self.token_counts = {
            "S0": 6,
            "U1": 5,
            "A1": 7,
            "U2": 5,
            "A2": 7,
        }

    def count_tokens(self, text: str) -> int:
        if text not in self.token_counts:
            raise ValueError(
                f"No predefined token count for: {text!r}"
            )

        return self.token_counts[text]

    def __call__(self, text: str) -> int:
        return self.count_tokens(text)



class TokenCounter:
    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        encoding_name: str = "o200k_base",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(
        self,
        prompt: str,
        backend: str = "ollama"
    ) -> int:

        if backend == "tiktoken":
            return len(self.encoding.encode(prompt))

        if backend == "ollama":
            return self._count_ollama_tokens(prompt)

        raise ValueError(
            f"Unsupported token counting backend: {backend}"
        )

    def _count_ollama_tokens(self, prompt: str) -> int:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "num_predict": 1,
                "num_ctx": 4096,
            },
        }

        request = Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urlopen(request, timeout=120) as response:
            result = json.load(response)

        if "prompt_eval_count" not in result:
            raise RuntimeError(
                "Ollama did not return prompt_eval_count"
            )

        return result["prompt_eval_count"]
