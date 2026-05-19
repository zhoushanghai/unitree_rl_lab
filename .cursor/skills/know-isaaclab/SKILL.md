---
name: know-isaaclab
description: Check official Isaac Lab documentation before proposing Isaac Lab settings, APIs, or implementation details. Use when the task mentions Isaac Lab configuration, markers, sensors, environment setup, or simulation workflow.
---

# Know Isaac Lab

## Purpose

When a request involves Isaac Lab usage, always verify relevant details from the official docs first, then provide guidance or code changes.

Official docs entry:
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/isaac_lab_tutorials/index.html

## Trigger Conditions

Apply this skill when the user asks about:
- Isaac Lab config fields or task/env settings
- Marker/visualization drawing
- Sensors, robot setup, simulation workflow, deployment
- API behavior that may vary by Isaac Sim/Isaac Lab version

## Required Workflow

1. Parse the request and extract 2-5 search keywords (for example: `marker`, `visualization`, `task config`).
2. Search the official documentation first (prefer `site:docs.isaacsim.omniverse.nvidia.com` scoped search).
3. Read at least one relevant official page before answering.
4. In the final response, include:
   - What was confirmed from docs
   - The source URL(s)
   - Any version caveat (for example, Isaac Sim 4.5.0 specific behavior)
5. If official docs do not clearly cover the question, state that explicitly, then provide the safest fallback suggestion.

## Marker Example

For requests like "how to draw marker":
- First look for Isaac Lab/Isaac Sim visualization or marker-related docs.
- Confirm the recommended API or workflow from official docs.
- Then provide project-specific implementation guidance.

## Output Style

- Keep answers concise and practical.
- Prefer doc-backed conclusions over assumptions.
- Do not invent unsupported Isaac Lab settings.
