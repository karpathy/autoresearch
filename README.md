# theRunner - Product R&D Agent

**theRunner** is an autonomous Product Research & Development Agent framework designed for high-reasoning models like Hermes. It transforms an AI agent into a dedicated researcher, strategist, and roadmapper for multiple product projects.

## How it works

The agent (theRunner) operates on projects defined in the `projects/` directory. It uses internet scraping to gather market intelligence and iteratively builds out product strategies and development roadmaps.

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

- **`theRunner.md`**: The core instructions and autonomous loop for the agent.
- **`scrape.py`**: A utility script for internet search and content extraction.
- **`pyproject.toml`**: Project dependencies.

## Quick Start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

1.  **Install dependencies**:
    ```bash
    uv sync
    ```

2.  **Add a new project**:
    Create a folder in `projects/` and add a `brief.md` describing your vision.

3.  **Boot theRunner**:
    Point your Hermes agent (or any LLM) to `theRunner.md` and let it start the autonomous research loop.

## Tools

theRunner uses `scrape.py` for internet access:

```bash
# Search for market data
uv run python scrape.py search "AI tutoring market trends 2025"

# Scrape a specific URL
uv run python scrape.py get https://techcrunch.com/article-about-competition
```

## License

MIT
