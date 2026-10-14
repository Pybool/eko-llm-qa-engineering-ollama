# LLM Quality Engineering Lab — Modules 01–08

A hands-on QA curriculum: understand LLM mechanisms, trigger realistic failure modes, and turn each finding into an automated regression test.

## One-time setup (Windows 11, PowerShell)

Install Git, VS Code and uv if you don't already have them:

```powershell
winget install --id Git.Git -e
winget install --id Microsoft.VisualStudioCode -e
winget install --id astral-sh.uv -e
```

Open a **new** PowerShell terminal:

```powershell
uv python install 3.12
cd path\to\llm-qa-engineering-lab
uv sync
uv run python -m llmqa.token_demo
uv run pytest -q
```

`uv sync` creates a Python environment and installs the Module 01 packages. `uv run` uses that project's environment automatically; there is no need to activate it manually.

To inspect code in VS Code: `code .`

## Our progressive curriculum

| Module | Focus | Practical deliverable |
|---|---|---|
| 01 | Tokenization, budgets, truncation | Token-budget analyzer and boundary tests |
| 02 | Transformer inference and decoding | Output consistency and decoding experiment |
| 03 | Context assembly and hierarchy | Context confusion and instruction-retention tests |
| 04 | Compaction and memory | Long-context state-retention regression suite |
| 05 | RAG | Retrieval, grounding and citation failure tests |
| 06 | Tools and agent loops | Tool-use state-machine and recovery tests |
| 07 | AI security | Prompt-injection and unauthorized-action test suite |
| 08 | Automated evaluation and CI/CD | Golden datasets, calibrated metrics and pipeline gates |

We will add implementation and tests to this **same project** module by module. Only Module 01 is implemented in this starter kit. Later modules are deliberately empty until you understand the preceding material.

## Module 01

Read `modules/01-tokenization/LESSON.md` and `modules/01-tokenization/CHALLENGES.md`.

Run:

```powershell
uv run python -m llmqa.token_demo
uv run pytest -q
```

Then deliberately expose two bugs:

```powershell
uv run pytest -q exercises/test_find_the_bugs.py
```

These exercise tests are **supposed to fail**. Repair `src/llmqa/unsafe_budget.py` so they pass, and then add cases for Unicode, text with no spaces, output reservation, and the exact boundary. Compare your proposed implementation to the safe, injectable counter in `src/llmqa/token_budget.py`.

## Important accuracy limitations

- `tiktoken` is for OpenAI tokenization. **It does not count Claude tokens exactly.** Use the Claude API token-count endpoint to measure a real Claude request when we introduce API tests.
- Tokenizing content strings is not equivalent to counting an entire API request, because messages, tools, images and other inputs have framing or model-specific costs. `token_budget.py` deliberately models overhead as an assumption.
- A token budget test does not prove an LLM answer is correct; it only checks one infrastructure risk.
- Everything in Module 01 runs without an API key or GPU.

## Safe project practices

Keep credentials only in local `.env` files (gitignored). Do not commit keys or production personal information. Use synthetic datasets in test cases.
