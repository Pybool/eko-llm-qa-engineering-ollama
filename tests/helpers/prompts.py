def generate_conversation_prompts() -> list[str]:
    return [
        "How many states and local governments are in Nigeria?",
        "Who are the current presidents of Nigeria and Ghana?",
        "Explain how Nigeria's banking system works.",
        "What happens when an interbank transfer fails?",
        "Explain transaction idempotency with an example.",
        "How do banks reconcile transactions?",
        "Explain double-entry bookkeeping in detail.",
        "Describe five common causes of failed payments.",
        "What security measures protect banking APIs?",
        "Summarize everything we have discussed so far.",
    ]

def generate_oversized_prompt(token_counter, history=None) -> str:
    """
    Generate a realistic financial audit prompt that exceeds
    ContextLab's configured token budget.
    """

    introduction = """
You are a financial auditor.

Analyze every transaction below.
Identify duplicate transactions, suspicious amounts,
failed payments and inconsistencies.

Do not omit any transaction from your analysis.

TRANSACTION RECORDS:
"""

    conclusion = """
Produce a complete audit report with:
1. Total transactions processed
2. Total successful payments
3. Total failed payments
4. Suspicious transactions
5. Detailed findings for every transaction
"""

    records = []

    for i in range(1, 251):
        record = f"""
Transaction ID: TXN-{i:06d}
Customer: Customer-{i}
Account Number: 100{i:07d}
Amount: NGN {i * 1375}.00
Status: {"FAILED" if i % 7 == 0 else "SUCCESS"}
Bank: Nigerian Commercial Bank
Description: Interbank transfer for invoice INV-{i:05d}
Reference: PAYMENT-{i * 98761}
"""
        records.append(record)

        # Check the budget after every 10 transactions.
        if i % 10 != 0:
            continue

        prompt = introduction + "\n".join(records) + conclusion

        # Respect the public API's 50,000-character limit.
        if len(prompt) > 50_000:
            break

        result = token_counter.count_tokens(
            prompt=prompt,
            history=history,
        )

        budget = result["budget"]

        print(
            f"Transactions: {i} | "
            f"Projected: {budget['projectedTotal']} | "
            f"Limit: {budget['contextLimit']} | "
            f"Remaining: {budget['remainingTokens']}"
        )

        if not budget["allowed"]:
            print(f"Context limit exceeded with {i} transactions.")
            return prompt

    raise AssertionError(
        "Unable to exceed the token budget " "within the configured test limits."
    )
