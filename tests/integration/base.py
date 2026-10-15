import requests


def fetch_conversation_chat_history(authenticated_client, conversation_id):
    response = authenticated_client.request("GET", f"/chats/{conversation_id}")
    assert response.status_code == 200
    history_data = response.json()    
    return history_data
        
def create_conversation(authenticated_client):
    payload = { "title": "QA Test Conversation" }
    response = authenticated_client.request(
        "POST",
        "/chats",
        json=payload
    )
    
    assert response.status_code == 201
    data = response.json()
    return (data, payload)


def create_message(authenticated_client, conversation_id, prompt):
    payload = { "content": prompt }
    response = authenticated_client.request(
        "POST",
        f"/chats/{conversation_id}/messages",
        json=payload
    )
    
    if response.status_code == 413:
        assert response.json()["error"]["code"] == "CONTEXT_LIMIT_EXCEEDED"
        assert response.json()["error"]["details"]["budget"]["allowed"] == False
        assert response.json()["error"]["details"]["budget"]["remainingTokens"] < 0
        
        return "ERRORED"
    
    return "PASSED"
    


def get_public_token_count_api(prompt):
    response = requests.post(
        "http://localhost:4000/api/tokens/count",
        json={
            "prompt": prompt
        },
        timeout=30,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "qwen2.5:1.5b"
    assert data["promptTokens"] > 0
    assert data["estimatedInputTokens"] >= data["promptTokens"]

    budget = data["budget"]

    assert budget["projectedTotal"] == (
        budget["inputTokens"]
        + budget["overheadTokens"]
        + budget["reservedOutputTokens"]
    )
    
    return data
    
    
