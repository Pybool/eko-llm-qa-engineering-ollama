
from llmqa.budget_policy import BudgetPolicy, BudgetReason


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