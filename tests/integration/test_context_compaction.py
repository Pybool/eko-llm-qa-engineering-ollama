import time

from tests.helpers.ollama_stats import get_ollama_invocations
from tests.helpers.prompts import generate_conversation_prompts
from tests.integration.base import create_conversation, create_message, fetch_conversation_chat_history

#on server we set budget.overLimitAction to "compact" before running this test, we should use api later for this sha
def test_verify_llm_compaction_works_properly(authenticated_client):
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
        print("\nPrompt:==> ", prompt)
        print("Status:==> ", status)
        
        # We know from test_06 that the 9th prompt (What security measures protect banking APIs?) always fails with CONTEXT_LIMIT_EXCEEDED
        
        assert status == "PASSED"
        sucessfulPrompts.append(prompt)
        ollama_invocation_after = get_ollama_invocations(authenticated_client)
        assert ollama_invocation_after["index"] == ollama_invocation_before["index"] + 1