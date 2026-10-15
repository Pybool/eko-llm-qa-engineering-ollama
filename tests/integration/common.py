import requests, time

from tests.helpers.ollama_stats import get_ollama_invocations
from tests.helpers.prompts import generate_conversation_prompts

def test_authenticated_user_can_retrieve_chats(authenticated_client):
    response = authenticated_client.request("GET", "/chats")

    assert response.status_code == 200

    data = response.json()

    assert "chats" in data
    assert isinstance(data["chats"], list), "'chats' must be an array"
    
    if(len(data["chats"]) > 0):
        assert all(isinstance(chat, dict) for chat in data["chats"])


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
    
    
def test_create_conversation(authenticated_client):
    
    data, payload = create_conversation(authenticated_client)

    assert isinstance(data, dict)
    assert "chat" in data
    assert data["chat"]["id"]
    assert data["chat"]["title"] == payload["title"]


def test_tok_006_long_conversation_exceeds_budget_without_modifying_history(authenticated_client):
    prompts = generate_conversation_prompts() 
    data, _ = create_conversation(authenticated_client)
    conversation_id = data.get("chat").get("id")
    sucessfulPrompts = []
    ollama_invocation_before = None
    ollama_invocation_after = None
            
    for prompt in prompts :
        time.sleep(1)
        ollama_invocation_before = get_ollama_invocations(authenticated_client)
        status = create_message(authenticated_client, conversation_id, prompt)
        if status == "ERRORED":
            break
        sucessfulPrompts.append(prompt)
        ollama_invocation_after = get_ollama_invocations(authenticated_client)
        assert ollama_invocation_after["index"] == ollama_invocation_before["index"] + 1

    ollama_invocation_after = get_ollama_invocations(authenticated_client)
    
     # our ollama llm should not be called again when context is exhausted
    assert ollama_invocation_after["index"] == ollama_invocation_before["index"]
    assert ollama_invocation_after["timestamp"] == ollama_invocation_before["timestamp"]
    
    chat_history = fetch_conversation_chat_history(authenticated_client, conversation_id)
    assert chat_history["chat"]["id"] == conversation_id
    assert len(sucessfulPrompts) == sum(1 for m in chat_history["messages"] if m["role"] == "user")
    
    user_messages_only = [msg for msg in chat_history["messages"] if msg["role"] == "user"]

    for user_msg, successful_prompt in zip(user_messages_only, sucessfulPrompts):
        assert user_msg["role"] == "user"
        assert user_msg["content"] == successful_prompt



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
    
    
