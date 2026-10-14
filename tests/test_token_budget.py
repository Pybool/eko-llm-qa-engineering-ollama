import pytest

from llmqa.token_budget import Message, evaluate_budget


def count_characters(text: str) -> int:
    """Fake deterministic counter, to isolate and verify budget arithmetic."""
    return len(text)


def test_exact_budget_boundary_is_accepted():
    result = evaluate_budget(
        [Message("user", "abc")],
        context_window=10,
        reserved_output_tokens=4,
        token_counter=count_characters,
        overhead_per_message=2,
        fixed_overhead=1,
    )
    assert result.estimated_input_tokens == 3
    assert result.projected_total == 10
    assert result.remaining_tokens == 0
    assert result.fits


def test_one_unit_over_budget_is_rejected():
    result = evaluate_budget(
        [Message("user", "abcd")],
        context_window=10,
        reserved_output_tokens=4,
        token_counter=count_characters,
        overhead_per_message=2,
        fixed_overhead=1,
    )
    assert result.remaining_tokens == -1
    assert not result.fits


def test_output_reservation_is_counted():
    result = evaluate_budget(
        [Message("user", "hello")],
        context_window=6,
        reserved_output_tokens=2,
        token_counter=count_characters,
        overhead_per_message=0,
    )
    assert not result.fits


def test_message_overhead_applied_per_message():
    messages = [Message("system", "a"), Message("user", "b")]
    result = evaluate_budget(
        messages,
        context_window=10,
        reserved_output_tokens=0,
        token_counter=count_characters,
        overhead_per_message=3,
    )
    assert result.estimated_input_tokens == 2
    assert result.estimated_overhead_tokens == 6
    assert result.projected_total == 8


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_invalid_context_window(bad: int):
    with pytest.raises(ValueError):
        evaluate_budget([], context_window=bad, reserved_output_tokens=0, token_counter=count_characters)


def test_rejects_negative_output_reservation():
    with pytest.raises(ValueError):
        evaluate_budget([], context_window=100, reserved_output_tokens=-1, token_counter=count_characters)
