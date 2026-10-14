"""Inspect how one OpenAI tokenizer divides different types of text.

Run: uv run python -m llmqa.token_demo
This tokenizer is NOT Claude's tokenizer.
"""

import tiktoken

from llmqa.token_budget import Message, evaluate_budget

SAMPLES = {
    "English": "Please verify this customer's payment.",
    "Nigerian Pidgin": "Abeg check whether my transfer don enter.",
    "Yoruba": "Ẹ jọ̀ọ́, ṣàyẹ̀wò ìsanwó yìí.",
    "Currency": "₦1,050,000.00",
    "Code": "if amount_minor_units < 0: raise ValueError('invalid')",
    "Emoji": "💸✅🇳🇬",
    "JSON": '{"country":"Nigeria","dialingCode":"+234"}',
    "Whitespace": "Nigeria  +234\n\t Lagos",
}


def main() -> None:
    encoding = tiktoken.get_encoding("o200k_base")
    for name, sample in SAMPLES.items():
        test_history = [
            Message("system", "Never perform monetary arithmetic with floats."),
            Message("user", sample),
        ]

        budget = evaluate_budget(
            test_history,
            context_window=100,
            reserved_output_tokens=40,
            token_counter=lambda text: len(encoding.encode(text)),
            overhead_per_message=4,
            fixed_overhead=6,
        )

        print(f"\n{name}")
        print(f"Input tokens: {budget.estimated_input_tokens}")
        print(f"Projected total: {budget.projected_total}")
        print(f"Remaining: {budget.remaining_tokens}")
        print(f"Fits budget: {budget.fits}")


if __name__ == "__main__":
    main()
