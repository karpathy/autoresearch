"""
Research Hub - Central coordination for multi-agent autonomous research community.
This module manages the registry of research threads, tracks discoveries,
and assigns research directions to avoid duplication.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional
from pathlib import Path


@dataclass
class ResearchThread:
    """Represents a single research direction/branch in the community."""
    branch: str
    hypothesis: str
    description: str
    tags: list[str] = field(default_factory=list)
    status: str = "active"  # active, exhausted, completed
    author: str = "agent"
    parent: Optional[str] = None  # builds on which branch
    commit: Optional[str] = None
    val_bpb: Optional[float] = None
    improvement: Optional[float] = None
    peak_memory_gb: Optional[float] = None
    num_params_m: Optional[float] = None
    depth: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    key_findings: list[str] = field(default_factory=list)
    future_directions: list[str] = field(default_factory=list)
    related_branches: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ResearchThread':
        return cls(**data)


@dataclass
class ResearchCommunity:
    """Maintains the state of the entire research community."""
    threads: dict[str, ResearchThread] = field(default_factory=dict)
    frontier: list[str] = field(default_factory=list)  # active research directions
    knowledge_tags: dict[str, list[str]] = field(default_factory=dict)  # tag -> branch names
    
    def to_dict(self) -> dict:
        return {
            'threads': {k: v.to_dict() for k, v in self.threads.items()},
            'frontier': self.frontier,
            'knowledge_tags': self.knowledge_tags
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ResearchCommunity':
        threads = {k: ResearchThread.from_dict(v) for k, v in data.get('threads', {}).items()}
        return cls(
            threads=threads,
            frontier=data.get('frontier', []),
            knowledge_tags=data.get('knowledge_tags', {})
        )


class ResearchHub:
    """
    Central coordinator for the autonomous research community.
    Manages thread registry, discovers related work, and assigns directions.
    """
    
    def __init__(self, storage_path: str = ".research_community"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.community_file = self.storage_path / "community.json"
        self.community = self._load_community()
    
    def _load_community(self) -> ResearchCommunity:
        """Load existing community state from disk."""
        if self.community_file.exists():
            try:
                with open(self.community_file, 'r') as f:
                    data = json.load(f)
                return ResearchCommunity.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load community state: {e}")
        return ResearchCommunity()
    
    def _save_community(self):
        """Persist community state to disk."""
        with open(self.community_file, 'w') as f:
            json.dump(self.community.to_dict(), f, indent=2)
    
    def register_thread(self, thread: ResearchThread) -> str:
        """Register a new research thread/branch."""
        self.community.threads[thread.branch] = thread
        
        # Add to frontier if active
        if thread.status == "active" and thread.branch not in self.community.frontier:
            self.community.frontier.append(thread.branch)
        
        # Update knowledge tags
        for tag in thread.tags:
            if tag not in self.community.knowledge_tags:
                self.community.knowledge_tags[tag] = []
            if thread.branch not in self.community.knowledge_tags[tag]:
                self.community.knowledge_tags[tag].append(thread.branch)
        
        self._save_community()
        return thread.branch
    
    def update_thread(self, branch: str, **updates):
        """Update an existing thread with new findings."""
        if branch not in self.community.threads:
            raise ValueError(f"Thread {branch} does not exist")
        
        thread = self.community.threads[branch]
        for key, value in updates.items():
            if hasattr(thread, key):
                setattr(thread, key, value)
        thread.updated_at = datetime.now().isoformat()
        
        # Update frontier status
        if thread.status != "active":
            if branch in self.community.frontier:
                self.community.frontier.remove(branch)
        
        self._save_community()
    
    def find_related_work(self, tags: list[str], exclude: list[str] = None) -> list[ResearchThread]:
        """Find threads with similar tags."""
        exclude = exclude or []
        related = []
        
        for tag in tags:
            if tag in self.community.knowledge_tags:
                for branch in self.community.knowledge_tags[tag]:
                    if branch not in exclude and branch in self.community.threads:
                        thread = self.community.threads[branch]
                        if thread.status == "active":
                            related.append(thread)
        
        # Deduplicate and sort by recency
        seen = set()
        unique = []
        for t in related:
            if t.branch not in seen:
                seen.add(t.branch)
                unique.append(t)
        
        return sorted(unique, key=lambda x: x.updated_at, reverse=True)
    
    def get_thread(self, branch: str) -> Optional[ResearchThread]:
        """Get a specific thread by branch name."""
        return self.community.threads.get(branch)
    
    def get_all_threads(self) -> list[ResearchThread]:
        """Get all research threads."""
        return list(self.community.threads.values())
    
    def get_active_threads(self) -> list[ResearchThread]:
        """Get all active research threads."""
        return [t for t in self.community.threads.values() if t.status == "active"]
    
    def get_frontier(self) -> list[str]:
        """Get list of active research branches."""
        return self.community.frontier.copy()
    
    def assign_direction(self, agent_id: str, preferred_tags: list[str] = None) -> ResearchThread:
        """
        Assign a new research direction to an agent.
        Tries to find an unexplored direction or extend an active one.
        """
        preferred_tags = preferred_tags or []
        
        # First, try to extend existing active threads
        active = self.get_active_threads()
        
        # Try to find related work based on preferred tags
        if preferred_tags:
            related = self.find_related_work(preferred_tags)
            if related:
                # Extend the most promising related thread
                parent = related[0]
                return self._create_extension(parent, agent_id)
        
        # Otherwise, start a new direction
        return self._create_new_direction(agent_id, preferred_tags)
    
    def _create_extension(self, parent: ResearchThread, agent_id: str) -> ResearchThread:
        """Create a new thread that extends an existing one."""
        branch = f"{parent.branch}-{agent_id}"
        
        thread = ResearchThread(
            branch=branch,
            hypothesis=f"Building on {parent.branch}",
            description=f"Extends {parent.branch} with new experiments",
            tags=parent.tags.copy(),
            status="active",
            author=agent_id,
            parent=parent.branch,
            related_branches=[parent.branch]
        )
        
        self.register_thread(thread)
        return thread
    
    def _create_new_direction(self, agent_id: str, tags: list[str]) -> ResearchThread:
        """Create a brand new research direction."""
        timestamp = datetime.now().strftime("%m%d")
        branch = f"autoresearch/{timestamp}-{agent_id}"
        
        thread = ResearchThread(
            branch=branch,
            hypothesis="New exploration",
            description="Independent research direction",
            tags=tags,
            status="active",
            author=agent_id
        )
        
        self.register_thread(thread)
        return thread
    
    def contribute_findings(
        self,
        branch: str,
        val_bpb: float,
        commit: str,
        key_findings: list[str],
        future_directions: list[str],
        improvement: float = None,
        peak_memory_gb: float = None,
        num_params_m: float = None,
        depth: int = None,
        status: str = "active"
    ):
        """Record findings from an agent's research run."""
        self.update_thread(
            branch,
            val_bpb=val_bpb,
            commit=commit,
            key_findings=key_findings,
            future_directions=future_directions,
            improvement=improvement,
            peak_memory_gb=peak_memory_gb,
            num_params_m=num_params_m,
            depth=depth,
            status=status
        )
    
    def get_dashboard(self) -> dict:
        """Generate a summary dashboard of the research community."""
        threads = list(self.community.threads.values())
        
        # Sort by val_bpb (lower is better)
        sorted_by_perf = sorted(
            [t for t in threads if t.val_bpb is not None],
            key=lambda x: x.val_bpb
        )
        
        return {
            "total_threads": len(threads),
            "active_threads": len([t for t in threads if t.status == "active"]),
            "best_result": sorted_by_perf[0].to_dict() if sorted_by_perf else None,
            "top_5_results": [t.to_dict() for t in sorted_by_perf[:5]],
            "all_tags": list(self.community.knowledge_tags.keys()),
            "frontier_branches": self.community.frontier
        }


# Convenience functions for CLI usage
def main():
    """CLI for interacting with the research hub."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Research Hub CLI")
    parser.add_argument("command", choices=["register", "update", "query", "dashboard"])
    parser.add_argument("--branch", help="Branch name")
    parser.add_argument("--tags", help="Comma-separated tags")
    parser.add_argument("--hypothesis", help="Research hypothesis")
    args = parser.parse_args()
    
    hub = ResearchHub()
    
    if args.command == "register":
        thread = ResearchThread(
            branch=args.branch,
            hypothesis=args.hypothesis or "New exploration",
            description="",
            tags=args.tags.split(",") if args.tags else []
        )
        hub.register_thread(thread)
        print(f"Registered thread: {args.branch}")
    
    elif args.command == "dashboard":
        dash = hub.get_dashboard()
        print(json.dumps(dash, indent=2))
    
    elif args.command == "query":
        if args.tags:
            results = hub.find_related_work(args.tags.split(","))
            for t in results:
                print(f"- {t.branch}: {t.hypothesis}")


if __name__ == "__main__":
    main()
