
from dataclasses import dataclass
from enum import Enum


class BudgetReason(str, Enum):
    WITHIN_BUDGET = "WITHIN_BUDGET"
    CONTEXT_LIMIT_EXCEEDED = "CONTEXT_LIMIT_EXCEEDED"


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    allowed: bool
    projected_total: int
    remaining_tokens: int
    reason: BudgetReason


class BudgetPolicy:
    @staticmethod
    def _validate_positive_integer(name: str, value: int) -> None:
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"{name} must be a positive integer."
            )

    @staticmethod
    def _validate_non_negative_integer(name: str, value: int) -> None:
        if type(value) is not int or value < 0:
            raise ValueError(
                f"{name} must be a non-negative integer."
            )

    def evaluate(
        self,
        context_limit: int,
        input_tokens: int,
        overhead_tokens: int,
        reserved_output_tokens: int,
    ) -> BudgetDecision:

        self._validate_positive_integer(
            "context_limit", context_limit
        )
        self._validate_non_negative_integer(
            "input_tokens", input_tokens
        )
        self._validate_non_negative_integer(
            "overhead_tokens", overhead_tokens
        )
        self._validate_positive_integer(
            "reserved_output_tokens", reserved_output_tokens
        )

        projected_total = (
            input_tokens
            + overhead_tokens
            + reserved_output_tokens
        )

        remaining_tokens = context_limit - projected_total
        allowed = projected_total <= context_limit

        reason = (
            BudgetReason.WITHIN_BUDGET
            if allowed
            else BudgetReason.CONTEXT_LIMIT_EXCEEDED
        )

        return BudgetDecision(
            allowed=allowed,
            projected_total=projected_total,
            remaining_tokens=remaining_tokens,
            reason=reason,
        )
