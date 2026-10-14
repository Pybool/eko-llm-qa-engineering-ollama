"""Token-budget estimates for an LLM request.

Important: tokenizing *message text* is not the same as counting an entire
provider API request. Message framing, tools, images, and other payloads may
consume additional tokens. This lab makes that uncertainty explicit by
requiring a separately configurable overhead allowance.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class BudgetResult:
    context_window: int
    estimated_input_tokens: int
    reserved_output_tokens: int
    estimated_overhead_tokens: int

    @property
    def projected_total(self) -> int:
        return (
            self.estimated_input_tokens
            + self.reserved_output_tokens
            + self.estimated_overhead_tokens
        )

    @property
    def remaining_tokens(self) -> int:
        return self.context_window - self.projected_total

    @property
    def fits(self) -> bool:
        return self.remaining_tokens >= 0


def evaluate_budget(
    messages: Iterable[Message],
    *,
    context_window: int,
    reserved_output_tokens: int,
    token_counter: Callable[[str], int],
    overhead_per_message: int = 4,
    fixed_overhead: int = 0,
) -> BudgetResult:
    """Estimate whether messages fit a configured *illustrative* budget.

    This checks budget math, NOT a provider's actual context acceptance.
    Use a provider count-tokens API for provider-specific production tests.
    The overhead values are scenario assumptions, not universal constants.
    """
    if context_window <= 0:
        raise ValueError("context_window must be positive")
    if min(reserved_output_tokens, overhead_per_message, fixed_overhead) < 0:
        raise ValueError("output reservation and overheads must not be negative")

    message_list = list(messages)
    estimate = sum(token_counter(message.content) for message in message_list)
    if estimate < 0:
        raise ValueError("token_counter produced an invalid negative count")
    overhead = fixed_overhead + len(message_list) * overhead_per_message

    return BudgetResult(
        context_window=context_window,
        estimated_input_tokens=estimate,
        reserved_output_tokens=reserved_output_tokens,
        estimated_overhead_tokens=overhead,
    )
