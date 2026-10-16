# LLM Quality Engineering Lab

A personal QA project where I'm testing how LLM applications behave when their context window starts filling up, and what happens when they eventually run out of space, I'm planning to cover instruction retention, hallucinations, prompt injection and other LLM failure scenarios as I go.


The app I'm testing is here: [ContextLab LLM Engineering App](https://github.com/Pybool/context-lab-llm-engineering-app). It runs on Ollama.

## What I've been testing

Most of the work so far is around two server configurations:

**1. Reject on context limit**

When the conversation becomes too large, the server should reject the request instead of sending it to Ollama.

I'm checking that:
- The token budget is calculated correctly, including boundary cases.
- Requests exceeding the limit are rejected.
- Previous conversation history is not affected by the rejection.
- Ollama isn't called again for rejected requests.

**2. Compact on context limit**

Instead of rejecting, the server can summarise older messages to free up context space.

Here I'm testing that conversations can continue after compaction and checking how the server handles the growing history.

I've also added some tests around authentication, creating and retrieving conversations, token counting and tracking Ollama invocations.

I'm gradually adding DeepEval to check the quality of the compacted summaries, since just reducing token count doesn't mean important information was actually retained.

## Tools

Python, pytest, uv, Ollama, DeepEval and Requests.

## Running the tests

I'm using Python 3.12 and uv for dependency management.

```powershell
uv sync
uv run pytest -q
```

For the integration tests, the ContextLab app and Ollama need to be running locally. The server configuration also needs to match the scenario being tested (`reject` or `compact`).

## Still working on

This is still an ongoing project. I'm planning to cover instruction retention, hallucinations, prompt injection and other LLM failure scenarios as I go.

The main idea is to showcase my understanding of how these systems actually behave, find where things can go wrong and automate the tests so I can catch those issues again later.