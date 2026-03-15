"""Gut check that the LLM scoring patterns in SCORE_DOCS.md actually work."""

from pydantic import BaseModel, Field


class EssayScores(BaseModel):
    argument_structure: int = Field(description="1-10: logical flow, clear thesis, supporting evidence")
    prose_clarity: int = Field(description="1-10: readable, concise, no jargon without purpose")
    originality: int = Field(description="1-10: fresh perspective, avoids cliches")


SAMPLE_ESSAY = """
The bicycle is the most efficient vehicle ever invented. Per calorie of energy input,
a human on a bicycle travels further than any other animal or machine. Cars move faster
but burn orders of magnitude more energy per mile. Planes are worse still. The bicycle
is also the only vehicle that makes its operator healthier with use — every other form
of transport is a net negative on the body. Cities designed around bicycles rather than
cars would be quieter, cleaner, and more pleasant to live in. The main obstacle is not
engineering but politics.
"""

SYSTEM = "You are a writing evaluator. Score the essay on each dimension. Be rigorous."


def test_anthropic():
    print("--- Anthropic (client.messages.parse + output_format) ---")
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"{SYSTEM}\n\n{SAMPLE_ESSAY}"},
        ],
        output_format=EssayScores,
    )

    scores = response.parsed_output
    print(f"  argument_structure: {scores.argument_structure}")
    print(f"  prose_clarity:      {scores.prose_clarity}")
    print(f"  originality:        {scores.originality}")
    print(f"  raw dict:           {scores.model_dump()}")
    return scores


def test_openai():
    print("--- OpenAI (client.beta.chat.completions.parse + response_format) ---")
    from openai import OpenAI

    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": SAMPLE_ESSAY},
        ],
        response_format=EssayScores,
    )

    scores = completion.choices[0].message.parsed
    print(f"  argument_structure: {scores.argument_structure}")
    print(f"  prose_clarity:      {scores.prose_clarity}")
    print(f"  originality:        {scores.originality}")
    print(f"  raw dict:           {scores.model_dump()}")
    return scores


if __name__ == "__main__":
    test_anthropic()
    print()
    test_openai()
    print("\nBoth patterns work.")
