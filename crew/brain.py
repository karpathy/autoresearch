"""Crew brain for autonomous decision-making.

The brain is the decision engine. It uses an LLM to:
1. Plan experiments for tasks
2. Analyze experiment results
3. Decide what to study when idle
4. Review papers/news for relevance
5. Generate code modifications

The LLM acts as an advisor that responds with structured JSON.
The brain parses responses and executes them. Fallback mode works
without LLM using systematic grid search.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict
import json
import logging
from itertools import product

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ExperimentPlan:
    """Plan for running experiments."""
    experiments: List[Dict]  # [{parameters: {...}, rationale: "..."}]
    rationale: str           # Overall plan reasoning
    early_stop_condition: Optional[str] = None
    notes: str = ""

    @staticmethod
    def from_json(data: Dict):
        """Create from JSON response."""
        if isinstance(data, str):
            data = json.loads(data)
        return ExperimentPlan(
            experiments=data.get("experiments", []),
            rationale=data.get("overall_rationale", ""),
            early_stop_condition=data.get("early_stop_condition"),
            notes=data.get("notes", ""),
        )


@dataclass
class TaskResults:
    """Analysis of task results."""
    summary: str
    best_experiment: Dict
    findings: List[str]
    knowledge_entries: List[Dict]
    suggested_follow_ups: List[Dict]
    notification_severity: str

    @staticmethod
    def from_json(data: Dict):
        """Create from JSON response."""
        if isinstance(data, str):
            data = json.loads(data)
        return TaskResults(
            summary=data.get("summary", ""),
            best_experiment=data.get("best_experiment", {}),
            findings=data.get("findings", []),
            knowledge_entries=data.get("knowledge_entries", []),
            suggested_follow_ups=data.get("suggested_follow_ups", []),
            notification_severity=data.get("notification_severity", "info"),
        )


@dataclass
class StudyTopic:
    """A self-directed study topic."""
    title: str
    reason: str
    experiment_plan: Dict
    expected_value: str

    @staticmethod
    def from_json(data: Dict):
        """Create from JSON response."""
        if isinstance(data, str):
            data = json.loads(data)
        return StudyTopic(
            title=data.get("title", ""),
            reason=data.get("reason", ""),
            experiment_plan=data.get("experiment_plan", {}),
            expected_value=data.get("expected_value", ""),
        )


@dataclass
class CodeModification:
    """Code changes to apply to train.py."""
    changes: List[Dict]  # [{old_text, new_text, description}]
    explanation: str

    @staticmethod
    def from_json(data: Dict):
        """Create from JSON response."""
        if isinstance(data, str):
            data = json.loads(data)
        return CodeModification(
            changes=data.get("changes", []),
            explanation=data.get("explanation", ""),
        )


# ============================================================================
# LLM Client Abstraction
# ============================================================================

class LLMClient:
    """Abstracts different LLM providers (Anthropic, OpenAI, Ollama, etc.)."""

    def __init__(self, config):
        """Initialize LLM client from config."""
        self.config = config
        self.provider = getattr(config, 'provider', 'anthropic')
        self.model = getattr(config, 'model', 'claude-haiku-4-5-20251001')
        self.api_key = getattr(config, 'api_key', '')
        self.base_url = getattr(config, 'base_url', None)
        self._client = None

    def get_client(self):
        """Get or initialize the LLM client."""
        if self._client is not None:
            return self._client

        if self.provider == 'anthropic':
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic module not installed")
                return None
        elif self.provider == 'openai':
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai module not installed")
                return None
        else:
            logger.warning(f"Unsupported LLM provider: {self.provider}")
            return None

        return self._client

    def chat(self, prompt: str, response_format: str = "text") -> str:
        """Send a prompt and get a response."""
        client = self.get_client()
        if client is None:
            logger.error("LLM client not available")
            return ""

        try:
            if self.provider == 'anthropic':
                response = client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

            elif self.provider == 'openai':
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return ""

        return ""


# ============================================================================
# Crew Brain
# ============================================================================

class CrewBrain:
    """Autonomous decision engine using LLM."""

    def __init__(self, config):
        """Initialize the brain with configuration."""
        self.config = config
        self.llm = LLMClient(config)

    def plan_experiments(self, task, knowledge: List[Dict] = None) -> ExperimentPlan:
        """Plan experiments for a task.

        Args:
            task: Task object with experiment specs
            knowledge: Relevant knowledge entries (optional)

        Returns: ExperimentPlan with experiments to run
        """
        # Try LLM first
        if self.llm.get_client() is not None:
            plan = self._plan_with_llm(task, knowledge)
            if plan and plan.experiments:
                return plan

        # Fallback: systematic grid search
        return self._plan_with_grid_search(task)

    def _plan_with_llm(self, task, knowledge: List[Dict]) -> Optional[ExperimentPlan]:
        """Plan using LLM."""
        if not task.experiment:
            return None

        prompt = self._format_experiment_prompt(task, knowledge)

        response = self.llm.chat(prompt)
        if not response:
            return None

        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ExperimentPlan.from_json(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return None

    def _plan_with_grid_search(self, task) -> ExperimentPlan:
        """Fallback: systematic grid search."""
        if not task.experiment or not task.experiment.get('parameters'):
            return ExperimentPlan(
                experiments=[],
                rationale="No parameter space specified",
            )

        parameters = task.experiment['parameters']
        num_experiments = task.experiment.get('num_experiments', 10)

        # Convert to list format
        param_names = []
        param_values = []

        for name, value in parameters.items():
            param_names.append(name)
            if isinstance(value, list):
                param_values.append(value)
            else:
                param_values.append([value])

        # Generate grid
        experiments = []
        for combo in product(*param_values):
            params = dict(zip(param_names, combo))
            experiments.append({
                "parameters": params,
                "rationale": "Systematic grid search",
            })

        # Limit to num_experiments
        experiments = experiments[:num_experiments]

        return ExperimentPlan(
            experiments=experiments,
            rationale=f"Systematic grid search ({len(experiments)} experiments)",
        )

    def analyze_results(self, task, experiments: List) -> TaskResults:
        """Analyze experiment results.

        Args:
            task: The completed task
            experiments: List of ExperimentResult objects

        Returns: TaskResults with findings and follow-ups
        """
        # Try LLM first
        if self.llm.get_client() is not None:
            results = self._analyze_with_llm(task, experiments)
            if results:
                return results

        # Fallback: basic analysis
        return self._analyze_with_defaults(task, experiments)

    def _analyze_with_llm(self, task, experiments) -> Optional[TaskResults]:
        """Analyze using LLM."""
        prompt = self._format_analysis_prompt(task, experiments)

        response = self.llm.chat(prompt)
        if not response:
            return None

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return TaskResults.from_json(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse analysis response: {e}")

        return None

    def _analyze_with_defaults(self, task, experiments) -> TaskResults:
        """Fallback analysis."""
        # Find best experiment
        best = None
        best_metric = float('inf')

        for exp in experiments:
            if exp.success and exp.metric_value is not None:
                if exp.metric_value < best_metric:
                    best_metric = exp.metric_value
                    best = exp

        best_data = {}
        if best:
            improvement = ""
            if task.experiment and task.experiment.get('baseline'):
                baseline = task.experiment['baseline']
                improvement = f" ({(baseline - best.metric_value) / baseline * 100:.1f}% improvement)"
            best_data = {
                "index": best.index,
                "metric_value": best.metric_value,
                "parameters": best.parameters,
                "improvement": improvement,
            }

        return TaskResults(
            summary=f"Completed {len(experiments)} experiments. Best: {best_data.get('metric_value', 'N/A')}",
            best_experiment=best_data,
            findings=[],
            knowledge_entries=[],
            suggested_follow_ups=[],
            notification_severity="info",
        )

    def decide_study_topic(self, knowledge: List[Dict], recent_tasks: List) -> StudyTopic:
        """Decide what to study when idle.

        Args:
            knowledge: All knowledge entries
            recent_tasks: Recent completed tasks

        Returns: StudyTopic to investigate
        """
        if self.llm.get_client() is not None:
            topic = self._study_with_llm(knowledge, recent_tasks)
            if topic:
                return topic

        # Fallback
        return StudyTopic(
            title="Explore hyperparameter interactions",
            reason="No specific study direction determined",
            experiment_plan={},
            expected_value="TBD",
        )

    def _study_with_llm(self, knowledge, recent_tasks) -> Optional[StudyTopic]:
        """Decide study topic using LLM."""
        prompt = f"""You are an autonomous ML crew member. The task board is empty.
You have free time to study. What would be most valuable to investigate?

Recent tasks: {len(recent_tasks)}
Knowledge entries: {len(knowledge)}

Respond with JSON:
{{
  "title": "What to study",
  "reason": "Why this is valuable",
  "experiment_plan": {{"type": "exploration", "num_experiments": 5}},
  "expected_value": "What we hope to learn"
}}
"""
        response = self.llm.chat(prompt)
        if not response:
            return None

        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return StudyTopic.from_json(data)
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    def generate_modifications(self, parameters: Dict, train_py_content: str) -> CodeModification:
        """Generate code modifications for train.py.

        Args:
            parameters: Parameter values to set
            train_py_content: Current train.py content

        Returns: CodeModification with what to change
        """
        # Use simple regex replacement (fallback method)
        import re

        changes = []
        result_content = train_py_content

        for param_name, value in parameters.items():
            param_upper = param_name.upper()
            pattern = rf"({param_upper}\s*=\s*)([^\n]+)"
            match = re.search(pattern, result_content)

            if match:
                old_text = match.group(0)
                new_text = f"{match.group(1)}{value}"
                changes.append({
                    "old_text": old_text,
                    "new_text": new_text,
                    "description": f"Set {param_name} = {value}",
                })
                result_content = result_content.replace(old_text, new_text, 1)

        return CodeModification(
            changes=changes,
            explanation=f"Modified {len(changes)} parameter(s)",
        )

    # ========================================================================
    # Prompt Formatting Helpers
    # ========================================================================

    def _format_experiment_prompt(self, task, knowledge) -> str:
        """Format task for experiment planning prompt."""
        prompt = f"""You are an autonomous ML research crew member. Plan experiments for this task.

TASK: {task.title}
DESCRIPTION: {task.description or '(no description)'}

EXPERIMENT TYPE: {task.experiment.get('type', 'general') if task.experiment else 'general'}
PARAMETER SPACE: {task.experiment.get('parameters', {}) if task.experiment else {}}
BUDGET: {task.experiment.get('num_experiments', 10) if task.experiment else 10} experiments
BASELINE: {task.experiment.get('baseline', 'unknown') if task.experiment else 'unknown'}

HINTS FROM CAPTAIN:
{self._format_hints(task.hints) if task.hints else '(none)'}

Respond with JSON:
{{
  "experiments": [
    {{"parameters": {{"param": "value"}}, "rationale": "why"}},
  ],
  "overall_rationale": "overall reasoning",
  "early_stop_condition": null,
  "notes": ""
}}
"""
        return prompt

    def _format_analysis_prompt(self, task, experiments) -> str:
        """Format results for analysis prompt."""
        exp_table = "\n".join([
            f"  exp_{i.index}: val_bpb={i.metric_value}, params={i.parameters}"
            for i in experiments[:10]
        ])

        prompt = f"""You are an ML research crew member. Analyze these experiment results.

TASK: {task.title}
EXPERIMENTS ({len(experiments)} total):
{exp_table}

Respond with JSON:
{{
  "summary": "One paragraph summary",
  "best_experiment": {{"index": 0, "metric_value": 0.0, "parameters": {{}}}},
  "findings": [],
  "knowledge_entries": [],
  "suggested_follow_ups": [],
  "notification_severity": "info"
}}
"""
        return prompt

    def _format_hints(self, hints) -> str:
        """Format captain's hints."""
        if not hints:
            return "(none)"
        lines = [f"  - {h.get('text', '')}" for h in hints]
        return "\n".join(lines)
