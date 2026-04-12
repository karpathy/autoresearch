# Copilot Instructions

This repository is an autonomous research framework. For full agent context, conventions, and workflow structure, see the root [AGENTS.md](../AGENTS.md).

## Quick Reference

- **Agent definitions**: `agents/` directory
- **Skill definitions**: `skills/` directory  
- **Research workflows**: `workflows/` directory
- **Workflow template**: `workflows/_template/`
- **Scaffold new workflow**: `python scaffold.py <name>`

## Key Convention

The source of truth for agent and skill definitions lives in the top-level `agents/` and `skills/` directories. This file and the `.github/` directory contain integration pointers only, not duplicated definitions.
