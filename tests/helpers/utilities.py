import re
import time
from litellm.exceptions import RateLimitError

def set_budget_policy(authenticated_client, action: str):
    response = authenticated_client.request(
        "POST",
        "/qa/set-budget-over-limit-action",
        json={"overLimitAction": action},
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["success"] is True
    assert data["budget"]["overLimitAction"] == action

    return data



def set_context_window(
    authenticated_client,
    policy_limit_tokens: int | None = None,
    context_window_tokens: int | None = None,
):
    payload = {}

    if policy_limit_tokens is not None:
        payload["policyLimitTokens"] = policy_limit_tokens

    if context_window_tokens is not None:
        payload["contextWindowTokens"] = context_window_tokens

    if not payload:
        raise ValueError(
            "At least one context window setting is required."
        )

    response = authenticated_client.request(
        "POST",
        "/qa/set-context-window",
        json=payload,
    )

    assert response.status_code == 200, response.text

    data = response.json()

    assert data["success"] is True

    if policy_limit_tokens is not None:
        assert (
            data["budget"]["policyLimitTokens"]
            == policy_limit_tokens
        )

    if context_window_tokens is not None:
        assert (
            data["modelContextWindowTokens"]
            == context_window_tokens
        )

    return data


def message_joiner(messages: list[dict], through_message_id: str | None = None) -> str:
    if through_message_id is not None:
        index = next(
            i
            for i, message in enumerate(messages)
            if message["id"] == through_message_id
        )
        messages = messages[: index + 1]

    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def evaluate_with_retry(
    metric,
    test_case,
    max_attempts: int = 5,
) -> float:

    for attempt in range(1, max_attempts + 1):
        try:
            return metric.measure(test_case)

        except RateLimitError as exc:
            if attempt == max_attempts:
                raise

            match = re.search(
                r"Please try again in\s*([\d.]+)\s*(ms|s)\b",
                str(exc),
                re.IGNORECASE,
            )

            if match:
                value = float(match.group(1))
                unit = match.group(2).lower()

                wait_seconds = (
                    value / 1000 if unit == "ms" else value
                )

                # Add a small safety margin
                wait_seconds += 2
            else:
                # Exponential backoff when no wait time is supplied
                wait_seconds = min(60, 5 * (2 ** (attempt - 1)))

            print(
                f"Groq rate limit reached. "
                f"Retrying in {wait_seconds:.1f}s "
                f"(attempt {attempt}/{max_attempts})"
            )

            time.sleep(wait_seconds)

    raise RuntimeError("Judge evaluation retries exhausted.")