import tiktoken

from llmqa.budget_policy import BudgetPolicy, BudgetReason
from llmqa.ollama_token_counter import TokenCounter, FakeTokenCounter
from src.llmqa.token_budget import Message, evaluate_budget


def test_tok_001_request_within_budget():
    """
    TOK-001: Verify that a request comfortably, within the context window is accepted.
    """

    context_limit = 100
    input_tokens = 20
    overhead_tokens = 10
    reserved_output_tokens = 20

    policy = BudgetPolicy()

    decision = policy.evaluate(
        context_limit=context_limit,
        input_tokens=input_tokens,
        overhead_tokens=overhead_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    
    assert decision.allowed is True
    assert decision.projected_total == 50
    assert decision.remaining_tokens == 50
    assert decision.reason == BudgetReason.WITHIN_BUDGET


def test_tok_002_request_at_budget():
    """
    TOK-002: Verify that a request Exactly at the Context Limit context window is accepted.
    """
    context_limit = 100
    input_tokens = 25
    overhead_tokens = 40
    reserved_output_tokens = 35

    policy = BudgetPolicy()
    
    decision = policy.evaluate(
        context_limit=context_limit,
        input_tokens=input_tokens,
        overhead_tokens=overhead_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    
    assert decision.allowed is True
    assert decision.projected_total == 100
    assert decision.remaining_tokens == 0
    assert decision.reason == BudgetReason.WITHIN_BUDGET
    
def test_tok_003_request_outside_budget():
    """
    TOK-003: Verify that a request exceeds the Context Limit context window is rejected.
    """

    context_limit = 100
    input_tokens = 25
    overhead_tokens = 41
    reserved_output_tokens = 35

    policy = BudgetPolicy()
    
    decision = policy.evaluate(
        context_limit=context_limit,
        input_tokens=input_tokens,
        overhead_tokens=overhead_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    
    assert decision.allowed is False
    assert decision.reason == BudgetReason.CONTEXT_LIMIT_EXCEEDED


def test_tok_004_same_characters_different_tokens():
    """
    TOK-004: Verify that prompts with equal character counts
    are validated using token counts, not character counts.
    """

    # Arrange
        
    counter = TokenCounter()

    prompt_a = "a" * 40
    prompt_b = "💸" * 40 # i expect this emoji to be higher, so we are assiming this

    # Verify both prompts have equal character lengths
    assert len(prompt_a) == len(prompt_b)

    # Count actual tokens
    tokens_a = counter.count_tokens(
            prompt_a, backend="tiktoken"
        )
    tokens_b = counter.count_tokens(
            prompt_b, backend="tiktoken"
        )

    assert tokens_b > tokens_a

    overhead_tokens = 10
    reserved_output_tokens = 20

    # we wiill Create a budget that fits prompt A exactly
    context_limit = (
        tokens_a + overhead_tokens + reserved_output_tokens
    )

    policy = BudgetPolicy()

    decision_a = policy.evaluate(
        context_limit=context_limit,
        input_tokens=tokens_a,
        overhead_tokens=overhead_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )

    decision_b = policy.evaluate(
        context_limit=context_limit,
        input_tokens=tokens_b,
        overhead_tokens=overhead_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )

    assert decision_a.allowed is True
    assert decision_a.reason == BudgetReason.WITHIN_BUDGET

    assert decision_b.allowed is False
    assert decision_b.reason == BudgetReason.CONTEXT_LIMIT_EXCEEDED

    assert decision_a.projected_total == context_limit
    assert decision_b.projected_total > context_limit

def test_tok_005_reserved_output_over_limit():
    """
    TOK-005: Verify that prompts with equal character counts
    are validated using token counts, not character counts.
    """
    
    context_limit = 100
    input_tokens = 60
    overhead_tokens = 10
    reserved_output_tokens = 20
    
    policy = BudgetPolicy()
    
    decision = policy.evaluate(
        context_limit=context_limit,
        input_tokens=input_tokens,
        overhead_tokens=overhead_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    
    expected_total = (input_tokens + overhead_tokens + reserved_output_tokens)
    
    assert decision.allowed is True
    assert decision.projected_total == expected_total
    assert decision.remaining_tokens == context_limit - expected_total
    assert decision.reason == BudgetReason.WITHIN_BUDGET
    
    reserved_output_tokens = 40
    
    decision = policy.evaluate(
            context_limit=context_limit,
            input_tokens=input_tokens,
            overhead_tokens=overhead_tokens,
            reserved_output_tokens=reserved_output_tokens,
        )
    
    expected_total = (input_tokens + overhead_tokens + reserved_output_tokens)
    
    assert decision.allowed is False
    assert decision.projected_total == expected_total
    assert decision.remaining_tokens == context_limit - expected_total
    assert decision.reason == BudgetReason.CONTEXT_LIMIT_EXCEEDED

