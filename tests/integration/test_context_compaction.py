import time

from deepeval import assert_test
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from tests.helpers.ollama_stats import get_ollama_invocations
from tests.helpers.utilities import message_joiner

from tests.helpers.prompts import generate_conversation_prompts
from tests.integration.base import create_conversation, create_message

def test_verify_llm_compaction_works_properly(authenticated_client, judge_model):
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
        # assert ollama_invocation_after["index"] == ollama_invocation_before["index"] + 1
    
    judge_context_compaction(authenticated_client, judge_model, conversation_id)


def judge_context_compaction(
    authenticated_client,
    judge_model,
    chat_id
):
    history_response = authenticated_client.request(
        "GET",
        f"/chats/{chat_id}",
    )

    assert history_response.status_code == 200

    messages = history_response.json()["messages"]

    context_response = authenticated_client.request(
        "GET",
        f"/qa/chats/{chat_id}/context",
    )

    assert context_response.status_code == 200

    compaction = context_response.json()["compaction"]

    assert compaction is not None
    
    through_id = compaction["throughMessageId"]

    index = next(
        i for i, message in enumerate(messages)
        if message["id"] == through_id
    )

    original_text = message_joiner(messages, through_id)
    
    print("chat_id =======================>", chat_id)

    test_case = LLMTestCase(
        input=original_text,
        actual_output=compaction["summary"],
    )

    metric = SummarizationMetric(
        threshold=0.8,
        model=judge_model,
        async_mode=False,
        truths_extraction_limit=10,
        include_reason=True,
    )

    assert_test(
        test_case,
        [metric],
    )
