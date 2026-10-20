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


def generate_mixed_conversation_prompts() -> list[str]:
    return [
        "What is the difference between machine learning and deep learning?",
        "How do neural networks learn from training data?",
        "Why do some AI models produce different answers to the same question?",
        "What are the main causes of the French Revolution?",
        "How do electric vehicles store and use energy?",
        "Why does the human body need sleep?",
        "What is the difference between REST and GraphQL?",
        "How does caching improve application performance?",
        "What happens when cached data becomes outdated?",
        "Why do some animals hibernate during winter?",
        "How are diamonds formed naturally?",
        "What makes cryptocurrencies different from traditional currencies?",
        "How do satellites maintain their orbit around Earth?",
        "What are the advantages of containerization in software development?",
        "Why do distributed systems sometimes experience data inconsistency?",
        "How does the human brain store long-term memories?",
        "What causes ocean tides?",
        "Explain how DNA replication works.",
        "What are the major challenges of colonizing Mars?",
        "Earlier, we discussed neural networks, caching, and distributed systems. Can you explain how these three concepts might work together in a large-scale AI application?",
    ]

def generate_hard_topics_conversation_prompts() -> list[str]:
    return [
        "Explain quantum computing and how it differs from classical computing.",
        "What are black holes, and what happens at the event horizon?",
        "Explain time dilation and why time passes differently depending on speed and gravity.",
        "How does quantum entanglement work, and why does it challenge our everyday understanding of physics?",
    ]
    

def generate_yoruba_answers_conversation_prompts() -> list[str]:
    return [
        "Tell me when a boy is supposed to start experiencing puberty?",
        "My younger brother forgot his school bag at home this morning.",
        "Explain why regular exercise and eating healthy food are important for the human body.",
        "The government announced that new roads and hospitals will be constructed next year.",
        "What are the major causes of unemployment among young people in Nigeria?",
        "Explain how commercial banks process money transfers between different financial institutions.",
        "Technology has completely changed how people communicate, work, and access information.",
        "Describe the advantages and disadvantages of using artificial intelligence in education.",
        "Scientists believe that climate change is causing more frequent floods and extreme temperatures.",
        "Explain the difference between classical computing and quantum computing.",
        "What happens when a massive star collapses and forms a black hole?",
        "Explain how time dilation affects astronauts travelling at speeds close to the speed of light.",
        "Describe how quantum entanglement allows particles to exhibit correlations across large distances.",
        "Although artificial intelligence can automate many repetitive tasks, human judgement remains important when making critical decisions.",
        "If a financial institution accidentally processes the same transaction twice, explain how an idempotency mechanism can prevent customers from being charged multiple times.",
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
