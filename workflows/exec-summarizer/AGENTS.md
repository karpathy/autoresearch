# exec-summarizer
Prompt optimization for executive news summaries. 4-dimension rubric via GPT-5.4-mini.

## Files
- `prompt.txt` | system prompt | agent-edited
- `evaluate.py` | scoring harness (Copilot SDK) | read-only
- `articles.json` | test dataset | read-only
- `workflow.yaml` | manifest | read-only
- `program.md` | strategy | read-only

## Rubric
- Conciseness | 2-3 sentences, no filler
- Executive Relevance | impact and "so what?" for decision-makers
- Source Provenance | data and quotes attributed with URIs
- ECQ Alignment | framing connects to Executive Core Qualifications

## Constraints
- Only modify prompt.txt -- evaluation integrity
- Prompt under 500 words -- brevity is the point
- Each eval takes 30-60s -- API call latency
