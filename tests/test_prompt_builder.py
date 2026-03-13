from __future__ import annotations

import unittest

from ttt_autoresearch.prompt_builder import build_rollout_prompt


class PromptBuilderTests(unittest.TestCase):
    def test_prompt_matches_current_rollout_contract(self) -> None:
        prompt = build_rollout_prompt(
            state_ctx="You are iteratively optimizing val_bpb.\nCurrent val_bpb (lower is better): 1.020000\nTarget: 0.970000",
            construction_section="unused construction",
            code_section="unused code",
        )
        self.assertIn("single-file language-model training script", prompt)
        self.assertIn("You are producing exactly one candidate patch", prompt)
        self.assertIn("## Objective", prompt)
        self.assertIn("## Fixed Project Facts", prompt)
        self.assertIn("## Non-Negotiable Constraints", prompt)
        self.assertIn("## Fixed Training / Evaluation Facts", prompt)
        self.assertIn("## Fixed Data Pipeline Contract", prompt)
        self.assertIn("## Fixed Model / Evaluation Contract", prompt)
        self.assertIn("## What You May Change", prompt)
        self.assertIn("## Technical Guidance", prompt)
        self.assertIn("## Budget-Aware Reasoning Requirement", prompt)
        self.assertIn("## What Good Edits Usually Look Like", prompt)
        self.assertIn("## What To Avoid", prompt)
        self.assertIn("## Output Requirements", prompt)
        self.assertIn("## Current Starting Point", prompt)
        self.assertIn("300 seconds of wall-clock training time", prompt)
        self.assertIn("maximum sequence length is fixed at `2048`", prompt)
        self.assertIn("evaluation token budget is fixed at `40 * 524288`", prompt)
        self.assertIn("validation is pinned to shard `06542`", prompt)
        self.assertIn("tokenizer vocabulary size is fixed at `8192`", prompt)
        self.assertIn("forward(x, y, reduction='none')", prompt)
        self.assertIn("TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0", prompt)
        self.assertIn("`DEVICE_BATCH_SIZE * MAX_SEQ_LEN`", prompt)
        self.assertIn('`WINDOW_PATTERN = "SSSL"` may be less efficient than `"L"`', prompt)
        self.assertIn("Return only exact search-and-replace patch blocks", prompt)
        self.assertIn("<search>", prompt)
        self.assertIn("</search>", prompt)
        self.assertIn("<replace>", prompt)
        self.assertIn("</replace>", prompt)
        self.assertIn("Current val_bpb (lower is better): 1.020000", prompt)
        self.assertIn("Target: 0.970000", prompt)
        self.assertNotIn("prepare.py", prompt)
        self.assertNotIn("pyproject.toml", prompt)
        self.assertNotIn("unused construction", prompt)
        self.assertNotIn("unused code", prompt)


if __name__ == "__main__":
    unittest.main()
