def test_public_token_count_api(token_counter):

    prompt = "Transfer 25000 NGN to another account."

    data = token_counter.count_tokens(prompt)

    assert data["model"] == "qwen2.5:1.5b"
    assert data["promptTokens"] > 0
    assert data["estimatedInputTokens"] >= data["promptTokens"]

    budget = data["budget"]

    assert budget["projectedTotal"] == (
        budget["inputTokens"]
        + budget["overheadTokens"]
        + budget["reservedOutputTokens"]
    )

def test_conversation_token_growth(token_counter):

    history = [
        {
            "role": "user",
            "content": "I want to transfer money."
        },
        {
            "role": "assistant",
            "content": "How much would you like to transfer?"
        },
    ]

    first = token_counter.count_tokens(
        prompt="25000 NGN"
    )

    second = token_counter.count_tokens(
        prompt="25000 NGN",
        history=history,
    )

    assert (
        second["estimatedInputTokens"]
        > first["estimatedInputTokens"]
    )