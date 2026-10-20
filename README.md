# LLM Quality Engineering Lab

An ongoing AI Quality Engineering project focused on testing LLM applications, identifying model failures, and building automated evaluations to detect regressions.

I'm using a separate application, [ContextLab LLM Engineering App](https://github.com/Pybool/context-lab-llm-engineering-app), powered by Ollama, as the system under test.

The goal is to evaluate both **application reliability** and **LLM behavior**, using API-level integration tests, deterministic assertions, and AI-assisted evaluations.

## Tools & Technologies

**Language:** Python 3.12

**Testing:** pytest, Requests, DeepEval

**LLM Infrastructure:** Ollama, Qwen, LiteLLM, Groq, Gemini, OpenAI

**Tokenization:** tiktoken, model-specific token counting

**Dependency Management:** uv


## Test Coverage

### 1. Token Budget & Context Window Management

Testing how the application manages token usage as conversations grow.

- Token counting, budget calculations, and boundary conditions.
- Context-limit rejection and HTTP error handling.
- Verification that rejected requests do not invoke Ollama.
- Conversation-history integrity after failed requests.
- Configurable context-window sizes and application token limits.

### 2. Conversation Compaction

Testing automatic context summarization when conversations approach their configured limits.

- Compaction triggers and token reduction.
- Conversation continuity after summarization.
- Preservation of original message history.
- Compaction snapshots and Ollama invocation tracking.
- Summary quality evaluation using DeepEval.
- Detection of missing information, altered facts, and unsupported claims.

### 3. Instruction Retention

Evaluating whether models continue following instructions across multiple conversation turns.

- Deterministic instruction compliance, including required response prefixes.
- Semantic instruction compliance, including response style and complexity.
- Multilingual instruction retention.
- Short- and long-conversation scenarios.
- Testing without compaction to isolate model behavior.
- Individual response evaluation using DeepEval's `PromptAlignmentMetric`.

### 4. API Integration Testing

Testing ContextLab's backend independently of its frontend.

- Authentication and authenticated API access.
- Conversation creation and retrieval.
- Message generation and persistence.
- Token-counting endpoints.
- Budget previews and overflow handling.
- QA configuration endpoints for switching between `reject` and `compact` policies.
- Dynamic context-window configuration and test-fixture cleanup.

## Evaluation Architecture

The project separates the **model being tested** from the **model evaluating its responses**.

**System under test:** ContextLab API → Ollama → Qwen

**Test execution:** Python → pytest → HTTP APIs → Assertions

**Semantic evaluation:** DeepEval → Configurable LLM judge → Evaluation scores and failure reasons

The evaluation layer supports Ollama, Gemini, OpenAI, and LiteLLM-compatible providers.

For Groq-based judges, I've implemented rate-aware execution with:

- Requests-per-minute (RPM) and tokens-per-minute (TPM) limits.
- Token estimation and usage reconciliation.
- Rate-limit handling and retries.
- Synchronous and asynchronous evaluation support.

This helps keep large evaluation runs within provider quotas and separates evaluation infrastructure failures from model-quality failures.

## Project Structure

```text
src/llmqa/
    Core QA utilities and evaluation model providers

tests/integration/
    API integration, context management,
    compaction and instruction-retention tests

tests/helpers/
    Authentication, token counting, prompt generation,
    API utilities and test configuration

tests/conftest.py
    Shared pytest fixtures and setup
```

The test suite uses reusable helpers and fixtures to reduce duplication and manage test prerequisites.

Tests that modify the application's context configuration can restore the default settings afterward, reducing configuration leakage between tests.

## Running the Tests

The integration tests require [ContextLab](https://github.com/Pybool/context-lab-llm-engineering-app) and Ollama to be running locally.

Install dependencies:

```powershell
uv sync
```

Create your local environment configuration:

```powershell
Copy-Item .env.example .env
```

Configure the evaluation provider and required credentials in `.env`.

Run the test suite:

```powershell
uv run pytest -q
```

Tests use environment-based judge configuration and ContextLab's QA endpoints to prepare specific context-limit scenarios.

Configuration-changing tests should run serially because they modify shared server settings.

## Findings & Ongoing Work

The project has already exposed several interesting LLM behaviors:

- Models can ignore instructions even when those instructions remain present in the conversation context.
- Compaction can successfully reduce token usage while losing important information.
- Models can accept incorrect user claims and repeat them as established facts.

These findings are helping shape additional evaluation scenarios.

I'm expanding the suite to cover hallucinations, factual contamination, sycophancy, prompt injection, adversarial inputs, and broader model-quality regressions.

The focus is on building practical, repeatable tests that reveal where LLM applications fail, help isolate the underlying cause, and make those failures easier to detect after future changes.