"""
GitHub Integration Utilities for Autoresearch Community
Provides functions for agents to interact with GitHub APIs for
discovering prior work and contributing findings.
"""

import os
import subprocess
import json
from typing import Optional
from dataclasses import dataclass


@dataclass
class GitHubDiscussion:
    """Represents a GitHub Discussion."""
    number: int
    title: str
    body: str
    category: str
    author: str
    created_at: str


@dataclass
class GitHubPR:
    """Represents a GitHub Pull Request."""
    number: int
    title: str
    body: str
    head_branch: str
    base_branch: str
    state: str
    author: str


class GitHubClient:
    """Lightweight GitHub API client using gh CLI."""
    
    def __init__(self, owner: str = "karpathy", repo: str = "autoresearch"):
        self.owner = owner
        self.repo = repo
        self._check_auth()
    
    def _check_auth(self):
        """Check if gh CLI is authenticated."""
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError("GitHub CLI not authenticated. Run 'gh auth login' first.")
    
    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run a gh command."""
        return subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True
        )
    
    def get_discussions(self, category: str = None, limit: int = 10) -> list[GitHubDiscussion]:
        """Fetch recent discussions."""
        args = ["api", f"repos/{self.owner}/{self.repo}/discussions"]
        
        if category:
            args[2] = f"repos/{self.owner}/{self.repo}/discussions/categories/{category}"
        
        result = self._run(args)
        
        if result.returncode != 0:
            print(f"Warning: Could not fetch discussions: {result.stderr}")
            return []
        
        try:
            data = json.loads(result.stdout)
            discussions = []
            
            for item in data[:limit]:
                discussions.append(GitHubDiscussion(
                    number=item['number'],
                    title=item['title'],
                    body=item['body'],
                    category=item.get('category', {}).get('name', 'General'),
                    author=item['author']['login'],
                    created_at=item['createdAt']
                ))
            return discussions
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse discussions: {e}")
            return []
    
    def create_discussion(
        self,
        title: str,
        body: str,
        category: str = "General"
    ) -> Optional[int]:
        """Create a new discussion (research paper)."""
        # First get category ID
        cat_result = self._run([
            "api", 
            f"repos/{self.owner}/{self.repo}/discussions/categories"
        ])
        
        if cat_result.returncode != 0:
            print(f"Warning: Could not get categories: {cat_result.stderr}")
            return None
        
        try:
            categories = json.loads(cat_result.stdout)
            cat_id = None
            for c in categories:
                if c['name'] == category:
                    cat_id = c['id']
                    break
            
            if not cat_id:
                cat_id = categories[0]['id']  # Default to first
        except (json.JSONDecodeError, KeyError):
            print("Warning: Could not parse categories")
            return None
        
        # Create discussion via GraphQL (gh api doesn't support discussion creation directly)
        query = """
        mutation($repoId: ID!, $catId: ID!, $title: String!, $body: String!) {
            createDiscussion(input: {
                repositoryId: $repoId,
                categoryId: $catId,
                title: $title,
                body: $body
            }) {
                discussion { number }
            }
        """
        
        # Get repo ID first
        repo_result = self._run([
            "api",
            "-f", f"query={{repository(owner:\"{self.owner}\", name:\"{self.repo}\"){{id}}}}"
        ])
        
        if repo_result.returncode != 0:
            print(f"Warning: Could not get repo ID: {repo_result.stderr}")
            return None
        
        try:
            repo_id = json.loads(repo_result.stdout)['data']['repository']['id']
        except (json.JSONDecodeError, KeyError):
            print("Warning: Could not parse repo ID")
            return None
        
        # Create discussion
        create_result = self._run([
            "api",
            "-f", f"query={query}",
            "-f", f"repoId={repo_id}",
            "-f", f"catId={cat_id}",
            "-f", f"title={title}",
            "-f", f"body={body}"
        ])
        
        if create_result.returncode != 0:
            print(f"Warning: Could not create discussion: {create_result.stderr}")
            return None
        
        try:
            return json.loads(create_result.stdout)['data']['createDiscussion']['discussion']['number']
        except (json.JSONDecodeError, KeyError):
            return None
    
    def get_pulls(self, state: str = "open", limit: int = 10) -> list[GitHubPR]:
        """Fetch pull requests."""
        result = self._run([
            "api",
            f"repos/{self.owner}/{self.repo}/pulls",
            "-f", f"state={state}",
            "-f", f"per_page={limit}"
        ])
        
        if result.returncode != 0:
            print(f"Warning: Could not fetch PRs: {result.stderr}")
            return []
        
        try:
            data = json.loads(result.stdout)
            prs = []
            
            for item in data:
                prs.append(GitHubPR(
                    number=item['number'],
                    title=item['title'],
                    body=item['body'],
                    head_branch=item['head']['ref'],
                    base_branch=item['base']['ref'],
                    state=item['state'],
                    author=item['user']['login']
                ))
            return prs
        except (json.JSONDecodeError, KeyError):
            print(f"Warning: Could not parse PRs: {e}")
            return []
    
    def create_pr(
        self,
        title: str,
        body: str,
        head: str,
        base: str = "master"
    ) -> Optional[int]:
        """Create a pull request (for sharing research branches)."""
        result = self._run([
            "api",
            f"repos/{self.owner}/{self.repo}/pulls",
            "-f", f"title={title}",
            "-f", f"body={body}",
            "-f", f"head={head}",
            "-f", f"base={base}"
        ])
        
        if result.returncode != 0:
            print(f"Warning: Could not create PR: {result.stderr}")
            return None
        
        try:
            data = json.loads(result.stdout)
            return data['number']
        except (json.JSONDecodeError, KeyError):
            return None
    
    def get_branches(self, limit: int = 50) -> list[str]:
        """Get list of branches."""
        result = self._run([
            "api",
            f"repos/{self.owner}/{self.repo}/branches",
            "-f", f"per_page={limit}"
        ])
        
        if result.returncode != 0:
            return []
        
        try:
            data = json.loads(result.stdout)
            return [b['name'] for b in data]
        except (json.JSONDecodeError, KeyError):
            return []


# Convenience functions for agents
def read_community_discussions(owner: str = "karpathy", repo: str = "autoresearch") -> list[GitHubDiscussion]:
    """Fetch and display recent community discussions."""
    client = GitHubClient(owner, repo)
    return client.get_discussions()


def read_community_prs(owner: str = "karpathy", repo: str = "autoresearch") -> list[GitHubPR]:
    """Fetch open PRs from the community."""
    client = GitHubClient(owner, repo)
    return client.get_pulls(state="all")


def contribute_paper(
    title: str,
    body: str,
    owner: str = "karpathy",
    repo: str = "autoresearch"
) -> Optional[int]:
    """Submit a research paper as a GitHub Discussion."""
    client = GitHubClient(owner, repo)
    return client.create_discussion(title, body, "Research Papers")


def share_branch(
    branch: str,
    description: str,
    owner: str = "karpathy",
    repo: str = "autoresearch"
) -> Optional[int]:
    """Share a research branch as a PR (without merging)."""
    client = GitHubClient(owner, repo)
    return client.create_pr(
        title=f"[Research] {branch}",
        body=description,
        head=branch
    )


if __name__ == "__main__":
    # Example: List community discussions
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "discussions":
        discussions = read_community_discussions()
        for d in discussions:
            print(f"#{d.number}: {d.title} ({d.category}) by {d.author}")
            print(f"   {d.body[:200]}...")
            print()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "prs":
        prs = read_community_prs()
        for pr in prs:
            print(f"#{pr.number}: {pr.title}")
            print(f"   Branch: {pr.head_branch} -> {pr.base_branch}")
            print(f"   Author: {pr.author}")
            print()
    
    else:
        print("Usage: python github_integration.py [discussions|prs]")
