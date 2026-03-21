"""
Research Swarm - Multi-agent execution framework for autonomous research.
This module manages running multiple research agents in parallel,
each exploring different research directions.
"""

import asyncio
import subprocess
import os
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable
from pathlib import Path
from research_hub import ResearchHub, ResearchThread


@dataclass
class AgentConfig:
    """Configuration for a research agent."""
    agent_id: str
    branch: str
    preferred_tags: list[str] = field(default_factory=list)
    max_experiments: int = 10
    time_budget_per_run: int = 330  # 5.5 minutes (including overhead)
    base_branch: str = "master"
    
    def to_dict(self) -> dict:
        return {
            'agent_id': self.agent_id,
            'branch': self.branch,
            'tags': self.preferred_tags,
            'max_experiments': self.max_experiments,
            'time_budget': self.time_budget_per_run,
            'base_branch': self.base_branch
        }


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    commit: str
    val_bpb: float
    training_seconds: float
    total_seconds: float
    peak_vram_mb: float
    mfu_percent: float
    total_tokens_m: float
    num_steps: int
    num_params_m: float
    depth: int
    status: str  # success, crash, timeout
    description: str = ""
    
    @classmethod
    def from_log(cls, log_content: str, commit: str, status: str = "success") -> 'ExperimentResult':
        """Parse experiment result from run.log content."""
        import re
        
        val_bpb = float(re.search(r'val_bpb:\s+([\d.]+)', log_content).group(1)) if re.search(r'val_bpb:\s+([\d.]+)', log_content) else 0.0
        training_seconds = float(re.search(r'training_seconds:\s+([\d.]+)', log_content).group(1)) if re.search(r'training_seconds:\s+([\d.]+)', log_content) else 0.0
        total_seconds = float(re.search(r'total_seconds:\s+([\d.]+)', log_content).group(1)) if re.search(r'total_seconds:\s+([\d.]+)', log_content) else 0.0
        peak_vram_mb = float(re.search(r'peak_vram_mb:\s+([\d.]+)', log_content).group(1)) if re.search(r'peak_vram_mb:\s+([\d.]+)', log_content) else 0.0
        mfu_percent = float(re.search(r'mfu_percent:\s+([\d.]+)', log_content).group(1)) if re.search(r'mfu_percent:\s+([\d.]+)', log_content) else 0.0
        total_tokens_m = float(re.search(r'total_tokens_M:\s+([\d.]+)', log_content).group(1)) if re.search(r'total_tokens_M:\s+([\d.]+)', log_content) else 0.0
        num_steps = int(re.search(r'num_steps:\s+(\d+)', log_content).group(1)) if re.search(r'num_steps:\s+(\d+)', log_content) else 0
        num_params_m = float(re.search(r'num_params_M:\s+([\d.]+)', log_content).group(1)) if re.search(r'num_params_M:\s+([\d.]+)', log_content) else 0.0
        depth = int(re.search(r'depth:\s+(\d+)', log_content).group(1)) if re.search(r'depth:\s+(\d+)', log_content) else 0
        
        return cls(
            commit=commit,
            val_bpb=val_bpb,
            training_seconds=training_seconds,
            total_seconds=total_seconds,
            peak_vram_mb=peak_vram_mb,
            mfu_percent=mfu_percent,
            total_tokens_m=total_tokens_m,
            num_steps=num_steps,
            num_params_m=num_params_m,
            depth=depth,
            status=status
        )


class ResearchAgent:
    """A single autonomous research agent."""
    
    def __init__(self, config: AgentConfig, hub: ResearchHub):
        self.config = config
        self.hub = hub
        self.experiments_run = 0
        self.best_val_bpb = float('inf')
        self.results = []
    
    async def run_experiment(self, description: str = "") -> ExperimentResult:
        """Run a single training experiment."""
        print(f"[{self.config.agent_id}] Running experiment {self.experiments_run + 1}")
        
        # Get current commit
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd="."
        ).stdout.strip()
        
        # Run training with timeout
        log_file = Path("run.log")
        
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    subprocess.run,
                    ["uv", "run", "train.py"],
                    capture_output=True,
                    text=True,
                    timeout=self.config.time_budget_per_run,
                    cwd="."
                ),
                timeout=self.config.time_budget_per_run + 60
            )
            
            # Parse results
            output = result.stdout + result.stderr
            
            if result.returncode == 0 and "val_bpb:" in output:
                exp_result = ExperimentResult.from_log(output, commit, "success")
            else:
                exp_result = ExperimentResult(commit=commit, val_bpb=0, training_seconds=0, 
                                               total_seconds=0, peak_vram_mb=0, mfu_percent=0,
                                               total_tokens_m=0, num_steps=0, num_params_m=0,
                                               depth=0, status="crash", description=description)
            
        except asyncio.TimeoutError:
            exp_result = ExperimentResult(commit=commit, val_bpb=0, training_seconds=0,
                                           total_seconds=0, peak_vram_mb=0, mfu_percent=0,
                                           total_tokens_m=0, num_steps=0, num_params_m=0,
                                           depth=0, status="timeout", description=description)
        
        self.experiments_run += 1
        self.results.append(exp_result)
        
        if exp_result.val_bpb > 0 and exp_result.val_bpb < self.best_val_bpb:
            self.best_val_bpb = exp_result.val_bpb
        
        return exp_result
    
    async def run_research_loop(self):
        """Main research loop for this agent."""
        print(f"[{self.config.agent_id}] Starting research on branch {self.config.branch}")
        
        while self.experiments_run < self.config.max_experiments:
            # Run experiment
            result = await self.run_experiment()
            
            # Record in hub
            improvement = None
            if len(self.results) > 1:
                prev = self.results[-2].val_bpb
                if prev > 0 and result.val_bpb > 0:
                    improvement = prev - result.val_bpb
            
            status = "active"
            if result.status == "crash" and self.experiments_run >= 3:
                status = "exhausted"
            
            self.hub.contribute_findings(
                branch=self.config.branch,
                val_bpb=result.val_bpb,
                commit=result.commit,
                key_findings=[f"Experiment {self.experiments_run}: {result.status}"],
                future_directions=[],
                improvement=improvement,
                peak_memory_gb=result.peak_vram_mb / 1024,
                num_params_m=result.num_params_m,
                depth=result.depth,
                status=status
            )
            
            # Decide: keep or discard based on val_bpb
            if result.status == "success" and len(self.results) >= 2:
                if result.val_bpb < self.results[-2].val_bpb:
                    # Improved - commit and continue
                    print(f"[{self.config.agent_id}] ✓ Improved! val_bpb: {result.val_bpb:.6f}")
                    self._git_commit(result.commit, f"improve: {result.val_bpb:.6f}")
                else:
                    # Did not improve - reset
                    print(f"[{self.config.agent_id}] ✗ No improvement, resetting")
                    self._git_reset()
        
        print(f"[{self.config.agent_id}] Research complete. Best: {self.best_val_bpb:.6f}")
    
    def _git_commit(self, message: str):
        """Commit changes to git."""
        subprocess.run(["git", "add", "-A"], capture_output=True)
        subprocess.run(["git", "commit", "-m", message], capture_output=True)
    
    def _git_reset(self):
        """Reset to parent commit."""
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], capture_output=True)


class ResearchSwarm:
    """
    Manages multiple research agents running in parallel.
    Coordinates agent assignments and aggregates results.
    """
    
    def __init__(self, num_agents: int = 4, hub: ResearchHub = None):
        self.num_agents = num_agents
        self.hub = hub or ResearchHub()
        self.agents: list[ResearchAgent] = []
    
    async def run_async(self, tags_per_agent: list[list[str]] = None) -> dict:
        """
        Run multiple agents in parallel, each exploring different directions.
        
        Args:
            tags_per_agent: Optional list of tags for each agent to focus on
        """
        tags_per_agent = tags_per_agent or [[] for _ in range(self.num_agents)]
        
        # Assign agents to research directions
        for i in range(self.num_agents):
            agent_id = f"agent{i:03d}"
            thread = self.hub.assign_direction(agent_id, tags_per_agent[i])
            
            config = AgentConfig(
                agent_id=agent_id,
                branch=thread.branch,
                preferred_tags=tags_per_agent[i]
            )
            
            agent = ResearchAgent(config, self.hub)
            self.agents.append(agent)
            
            # Create branch for this agent
            subprocess.run(["git", "checkout", "-b", thread.branch], capture_output=True)
        
        print(f"Running {self.num_agents} agents in parallel...")
        
        # Run all agents concurrently
        tasks = [agent.run_research_loop() for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        return self._aggregate_results()
    
    def _aggregate_results(self) -> dict:
        """Combine results from all agents."""
        all_results = []
        
        for agent in self.agents:
            all_results.extend(agent.results)
        
        # Find best result
        successful = [r for r in all_results if r.status == "success"]
        best = min(successful, key=lambda x: x.val_bpb) if successful else None
        
        return {
            "total_experiments": len(all_results),
            "successful": len(successful),
            "crashed": len([r for r in all_results if r.status == "crash"]),
            "best_val_bpb": best.val_bpb if best else None,
            "best_commit": best.commit if best else None,
            "agents": {
                a.config.agent_id: {
                    "experiments": a.experiments_run,
                    "best": a.best_val_bpb
                }
                for a in self.agents
            },
            "dashboard": self.hub.get_dashboard()
        }
    
    def run_sync(self, tags_per_agent: list[list[str]] = None) -> dict:
        """Synchronous wrapper for run_async."""
        return asyncio.run(self.run_async(tags_per_agent))


def generate_paper(agent_id: str, thread: ResearchThread, results: list[ExperimentResult]) -> str:
    """Generate a research contribution paper from agent results."""
    best = min([r for r in results if r.status == "success"], key=lambda x: x.val_bpb) if results else None
    
    paper = f"""# Research Contribution: {thread.branch}

## Hypothesis
{thread.hypothesis}

## Approach
{thread.description}

Tags: {', '.join(thread.tags)}

## Results
| Metric | Value |
|--------|-------|
| Best val_bpb | {best.val_bpb if best else 'N/A':.6f} |
| Total experiments | {len(results)} |
| Successful runs | {len([r for r in results if r.status == 'success'])} |
| Crashed runs | {len([r for r in results if r.status == 'crash'])} |

### Best Run Details
| Metric | Value |
|--------|-------|
| Commit | {best.commit if best else 'N/A'} |
| Training time | {best.training_seconds:.1f}s |
| Peak VRAM | {best.peak_vram_mb/1024:.1f}GB |
| Model params | {best.num_params_m:.1f}M |
| Depth | {best.depth} |

## Key Insights
"""
    
    for i, finding in enumerate(thread.key_findings, 1):
        paper += f"- {i}. {finding}\n"
    
    paper += """
## Future Directions
"""
    for i, direction in enumerate(thread.future_directions, 1):
        paper += f"- {i}. {direction}\n"
    
    paper += f"""
## Artifacts
- Branch: `{thread.branch}`
- Commit: `{thread.commit or 'N/A'}`
- Parent: `{thread.parent or 'None'}`

---
*Generated by {agent_id} on {datetime.now().isoformat()}*
"""
    
    return paper


# Example usage
if __name__ == "__main__":
    # Quick test
    hub = ResearchHub()
    
    # Register a baseline thread
    thread = ResearchThread(
        branch="autoresearch/baseline",
        hypothesis="Establish baseline performance",
        description="Initial baseline with default hyperparameters",
        tags=["baseline", "default"]
    )
    hub.register_thread(thread)
    
    # Show dashboard
    print(json.dumps(hub.get_dashboard(), indent=2))
