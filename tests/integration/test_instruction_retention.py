import time

from deepeval.metrics import PromptAlignmentMetric
from deepeval.test_case import LLMTestCase

from tests.helpers.prompts import (
    generate_mixed_conversation_prompts,
    generate_hard_topics_conversation_prompts,
    generate_yoruba_answers_conversation_prompts,
)
from tests.integration.base import (
    create_conversation,
    create_message,
    fetch_conversation_chat_history,
)

from tests.helpers.utilities import set_budget_policy, set_context_window


def _test_basic_instruction_retention_semantic(
    authenticated_client,
    judge_model,
):
    data, _ = create_conversation(authenticated_client)
    conversation_id = data["chat"]["id"]

    status = create_message(
        authenticated_client,
        conversation_id,
        get_basic_single_instruction_and_question_same_prompt(),
    )

    assert status == "PASSED"

    chat_history_data = fetch_conversation_chat_history(
        authenticated_client,
        conversation_id,
    )

    messages = chat_history_data["messages"]

    judge_prompt_instructions = [
        "The model must translate the sentence 'Why has the king forsaken me and my family?' into grammatically correct medieval English.",
        "The translation must preserve the original meaning and interrogative form of the sentence.",
        "The response must contain only the medieval English translation, without modern English explanations, introductions, or additional commentary.",
        "The model must translate the question rather than answer it.",
    ]

    assert_semantic_instruction_retention(
        messages,
        judge_model,
        judge_prompt_instructions,
        skip_initial_response=True
    )


def _test_short_term_instruction_retention_deterministic(
    authenticated_client,
):
    prompts = generate_mixed_conversation_prompts()

    data, _ = create_conversation(authenticated_client)
    conversation_id = data["chat"]["id"]

    status = create_message(
        authenticated_client,
        conversation_id,
        get_deterministic_instruction(),
    )

    assert status == "PASSED"

    for prompt in prompts:
        time.sleep(1)

        prompt_status = create_message(
            authenticated_client,
            conversation_id,
            prompt,
        )

        if prompt_status == "ERRORED":
            break

        assert prompt_status == "PASSED", (
            f"Prompt failed: {prompt}\n" f"Status: {prompt_status}"
        )

    chat_history_data = fetch_conversation_chat_history(
        authenticated_client,
        conversation_id,
    )

    assert_deterministic_instruction_retention(chat_history_data["messages"])


def _test_short_term_instruction_retention_semantic(
    authenticated_client,
    judge_model,
):
    prompts = generate_hard_topics_conversation_prompts()

    data, _ = create_conversation(authenticated_client)
    conversation_id = data["chat"]["id"]

    status = create_message(
        authenticated_client,
        conversation_id,
        get_semantic_instruction(),
    )

    assert status == "PASSED"

    for prompt in prompts:
        time.sleep(1)

        prompt_status = create_message(
            authenticated_client,
            conversation_id,
            prompt,
        )

        assert prompt_status == "PASSED", (
            f"Prompt failed: {prompt}\n" f"Status: {prompt_status}"
        )

    chat_history_data = fetch_conversation_chat_history(
        authenticated_client,
        conversation_id,
    )

    judge_prompt_instructions = [
        "Every explanation must use language understandable to a 10-year-old child.",
        "Every explanation must avoid unexplained technical jargon.",
        "Every explanation must include a relevant everyday-life analogy.",
    ]

    assert_semantic_instruction_retention(
        chat_history_data["messages"],
        judge_model,
        judge_prompt_instructions,
        skip_initial_response=True,
    )


def test_long_term_instruction_retention_semantic(
    authenticated_client, judge_model, restore_default_config
):
    set_budget_policy(authenticated_client, "reject")
    set_context_window(
        authenticated_client,
        policy_limit_tokens= 8192,
        context_window_tokens = 8192
    )
    prompts = generate_yoruba_answers_conversation_prompts()

    data, _ = create_conversation(authenticated_client)
    conversation_id = data["chat"]["id"]

    status = create_message(
        authenticated_client,
        conversation_id,
        get_yoruba_translation_instruction(),
    )

    assert status == "PASSED"

    for prompt in prompts:
        time.sleep(1)

        prompt_status = create_message(
            authenticated_client,
            conversation_id,
            prompt,
        )

        assert prompt_status == "PASSED", (
            f"Prompt failed: {prompt}\n" f"Status: {prompt_status}"
        )

    chat_history_data = fetch_conversation_chat_history(
        authenticated_client,
        conversation_id,
    )

    judge_prompt_instructions = [
        "Every assistant response must be written entirely in grammatically correct and natural Yoruba.",
        "The assistant must not respond in English or any other language.",
        "The assistant must preserve the meaning of the question or information being addressed.",
    ]

    assert_semantic_instruction_retention(
        chat_history_data["messages"],
        judge_model,
        judge_prompt_instructions,
        skip_initial_response=True
    )


def assert_deterministic_instruction_retention(
    messages: list[dict],
    expected_prefix: str = "~Eko",
) -> None:
    assistant_messages = [
        message for message in messages if message["role"] == "assistant"
    ]

    assert assistant_messages, "No assistant responses found."

    for index, message in enumerate(assistant_messages, start=1):
        content = message["content"]

        assert content.startswith(expected_prefix), (
            f"Instruction retention failed at assistant response #{index}\n"
            f"Message ID: {message['id']}\n"
            f"Expected prefix: {expected_prefix!r}\n"
            f"Actual response: {content[:150]!r}"
        )


def assert_semantic_instruction_retention(
    messages: list[dict],
    judge_model,
    judge_prompt_instructions: list[str],
    skip_initial_response: bool = False,
) -> None:
    assert messages, "Conversation history is empty."
    assert judge_prompt_instructions, "Judge instructions are required."

    turns = []
    initial_instruction = None
    current_user_prompt = None

    # Build user-assistant conversation turns
    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            if initial_instruction is None:
                initial_instruction = content

            current_user_prompt = content

        elif role == "assistant":
            assert (
                current_user_prompt is not None
            ), "Assistant response found without a corresponding user prompt."

            turns.append(
                {
                    "prompt": current_user_prompt,
                    "response": content,
                    "message_id": message.get("id"),
                }
            )

            current_user_prompt = None

    assert initial_instruction is not None
    assert turns, "No assistant responses found."

    # Optionally exclude the acknowledgment of the initial instruction
    if skip_initial_response:
        turns = turns[1:]

    assert turns, "No assistant responses available for evaluation."

    failures = []
    total = len(turns)

    for position, turn in enumerate(turns):
        response_number = position + (2 if skip_initial_response else 1)

        test_case = LLMTestCase(
            input=(
                f"Original conversation instruction:\n"
                f"{initial_instruction}\n\n"
                f"Current user prompt:\n"
                f"{turn['prompt']}"
            ),
            actual_output=turn["response"],
        )

        metric = PromptAlignmentMetric(
            prompt_instructions=judge_prompt_instructions,
            threshold=0.8,
            model=judge_model,
            async_mode=False,
            include_reason=True,
        )

        # Rate limiting and retries are handled by the judge provider.
        # Judge API failures must not be treated as semantic test failures.
        try:
            score = metric.measure(test_case)

        except Exception as exc:
            raise RuntimeError(
                f"LLM judge evaluation failed at assistant "
                f"response #{response_number} "
                f"(Message ID: {turn['message_id']}). "
                f"Evaluation stopped due to a judge error."
            ) from exc

        passed = metric.is_successful()

        status = "PASSED" if passed else "FAILED"

        print(
            f"\n[{position + 1}/{total}] "
            f"Assistant response #{response_number}\n"
            f"Message ID: {turn['message_id']}\n"
            f"Status: {status}\n"
            f"Score: {score:.2f} / 1.00\n"
            f"Threshold: {metric.threshold:.2f}\n"
            f"Reason: {metric.reason}\n"
        )

        if not passed:
            failures.append(
                {
                    "response_number": response_number,
                    "message_id": turn["message_id"],
                    "prompt": turn["prompt"],
                    "response": turn["response"],
                    "score": score,
                    "reason": metric.reason,
                }
            )

    # Produce one consolidated assertion after all evaluations
    if failures:
        failure_details = "\n\n".join(
            (
                f"Response #{failure['response_number']}\n"
                f"Message ID: {failure['message_id']}\n"
                f"Prompt: {failure['prompt']}\n"
                f"Score: {failure['score']:.2f}\n"
                f"Reason: {failure['reason']}\n"
                f"Response preview: {failure['response'][:300]}"
            )
            for failure in failures
        )

        raise AssertionError(
            f"Semantic instruction retention failed.\n"
            f"Evaluated: {total}\n"
            f"Passed: {total - len(failures)}\n"
            f"Failed: {len(failures)}\n\n"
            f"{failure_details}"
        )

    print(
        f"\nSemantic instruction retention PASSED.\n"
        f"All {total} assistant responses satisfied "
        f"the required alignment threshold."
    )


def get_basic_single_instruction_and_question_same_prompt() -> str:
    return (
        "Translate the following sentence into medieval English only. "
        "Do not answer the question or include a modern English explanation: "
        "'Why has the king forsaken me and my family?'"
    )


def get_yoruba_translation_instruction() -> str:
    return (
        "For the rest of this conversation, respond to every prompt "
        "in correct Yoruba language only. Do not use English or any "
        "other language."
    )


def get_deterministic_instruction() -> str:
    return (
        "For all answers to any prompt given, " "always prepend '~Eko' to the answers."
    )


def get_semantic_instruction() -> str:
    return (
        "For the rest of this conversation, explain every topic as if "
        "you are teaching a 10-year-old child. Use simple, "
        "beginner-friendly language, avoid complex technical jargon, "
        "and include a relatable everyday-life analogy in every explanation."
    )
