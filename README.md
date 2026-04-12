# Product R&D Agent

An autonomous swarm of agents (or just one very busy one) dedicated to product research, strategy, and development roadmapping.

## How it works

The agent operates on projects defined in the `projects/` directory. For each project, it performs market research using web scraping, synthesizes a product strategy, and develops a roadmap.

### Project Structure

```
projects/
  <project-name>/
    brief.md      - Initial vision (human-provided)
    research/     - Raw research data (agent-generated)
    strategy.md   - Product strategy (agent-generated)
    roadmap.md    - Development roadmap (agent-generated)
    status.md     - Current project status
```

### Core Components

- **`program.md`**: The brain of the agent. Contains instructions for the autonomous R&D loop.
- **`scrape.py`**: A utility script that allows the agent to search the web and extract content from websites.
- **`pyproject.toml`**: Project dependencies.

## Quick Start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

1.  **Install dependencies**:
    ```bash
    uv sync
    ```

2.  **Add a new project**:
    Create a folder in `projects/` and add a `brief.md`.

3.  **Run the agent**:
    Point your AI agent to `program.md` and let it start researching.

## Tools

The agent can use `scrape.py` for internet access:

```bash
# Search for something
uv run python scrape.py search "competitors for carbon tracking apps"

# Scrape a specific URL
uv run python scrape.py get https://example.com/article
```

## License

MIT
