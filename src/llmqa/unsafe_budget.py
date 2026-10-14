"""INTENTIONALLY BUGGY code for the red-team exercise. DO NOT use in production.

Your task: expose each bug with a failing test, then implement the fix.
"""


def count_tokens_badly(text: str) -> int:
    # BUG: a word is not a token.
    return len(text.split())


def can_send_badly(prompt: str, context_window: int, reserved_output_tokens: int) -> bool:
    # BUG: it completely ignores reserved_output_tokens.
    return count_tokens_badly(prompt) <= context_window
