# Restricted Retrieval RL Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal restricted-retrieval RL environment for `search`, `quote_evidence`, and `stop`, plus a rule-based rollout script and verification tests.

**Architecture:** Add a small `train_agent.rl` module that builds SciFact-backed episodes and exposes a deterministic environment with explicit rewards and termination. Keep this path independent from the current classifier trainer so the environment can be debugged in isolation.

**Tech Stack:** Python 3.8, standard library dataclasses, Hugging Face `datasets`, `unittest`

---
