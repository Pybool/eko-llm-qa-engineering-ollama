import time

from tests.helpers.ollama_stats import get_ollama_invocations
from tests.helpers.prompts import generate_conversation_prompts
from tests.integration.base import create_conversation, create_message, fetch_conversation_chat_history

#on server we set budget.overLimitAction to "reject"

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