# Agent Training Design

## Purpose

Build a trainable evidence-seeking agent framework for `deep_research_review_v2`. The framework should operate over a generic `claim` and optional `hypothesis`, gather supporting and contradicting evidence, update an evidence graph, decide the next action, and decide when to stop. Paper review is a core instance, but the training stack should remain dataset-agnostic and reusable.

## Architecture Options

### Option A: Single policy SFT

Train one causal LM to map full state directly to the next JSON action, including `stop` as one action type.

Pros:
- simplest implementation
- fastest to bootstrap from trajectory JSONL
- lowest engineering overhead

Cons:
- stopping behavior gets entangled with search behavior
- hard to evaluate action quality separately from stop quality
- difficult to plug in verifier or reranker modules later

### Option B: Multi-head staged agent

Use a shared state representation but train separate modules in stages:
- verifier / reranker
- next-action policy
- stop policy
- optional reviewer head for final writeup or critique

Pros:
- matches the target problem decomposition
- clean offline supervision path from heterogeneous datasets
- easier ablations and debugging
- verifier can improve both action selection and stopping later

Cons:
- more components to maintain
- requires consistent trajectory schema and interfaces

### Option C: Unified policy with post-hoc verifier reranking

Train one primary policy for next action and stopping, then apply a verifier or reranker only at inference time over sampled candidates.

Pros:
- inference quality can improve without retraining the policy
- lower initial training complexity than a fully decomposed agent

Cons:
- still mixes stopping and acting in one model
- reranker quality is limited by weak candidate generation
- makes later RL credit assignment less clear

## Recommendation

Recommend **Option B: Multi-head staged agent**.

Reasoning:
- the target system already decomposes naturally into search, graph update, verification, and stop control
- the requested training order maps directly to this setup
- it is the cleanest path for offline supervised learning first and optional RL later
- it supports claim-evidence tasks and review-specific tasks under one state/action format

## Problem Definition

### State

At step `t`, the agent state is a structured bundle:
- `claim`: normalized claim or review question
- `hypothesis`: current working hypothesis, optional
- `query_context`: paper metadata, topic metadata, benchmark-specific fields
- `history`: prior observations, chosen actions, retrieved documents, extracted evidence
- `evidence_graph`: nodes for claims, subclaims, evidence, sources, contradiction links, uncertainty tags
- `budget_state`: steps used, token budget, retrieval budget, optional time budget
- `verifier_state`: current support / contradiction / sufficiency signals

State should be serializable to JSONL and renderable to prompt text.

### Action

Base action space:
- `search`: issue a query for evidence or counter-evidence
- `open_document`: inspect a retrieved document or span
- `quote_evidence`: extract or record a useful support or contradiction snippet
- `update_graph`: add or revise evidence graph nodes / edges
- `ask_followup`: refine the search direction or subproblem
- `stop`: terminate evidence seeking and hand off to downstream head

Recommended later extensions:
- `rerank_candidates`
- `merge_duplicate_claims`
- `request_more_context`

### Reward

Initial training should not rely on online RL. Define offline reward targets for later use:
- positive for retrieving relevant support or contradiction evidence
- positive for graph updates that improve coverage or resolve uncertainty
- negative for redundant search, repeated evidence, unsupported graph updates
- positive for correct stopping under sufficient evidence
- negative for premature stop and over-search
- final task reward from benchmark outcome or review quality when available

### Stopping

Stopping is a separate decision problem. The stop policy predicts:
- `should_stop`: yes / no
- `reason`: sufficiency, conflict resolved, budget exhausted, evidence sparse, etc.

Stopping should depend on evidence sufficiency, contradiction coverage, and budget state, not only current observation text.

### Outputs

The integrated loop should produce:
- trajectory JSONL for training
- final evidence graph
- structured evidence summary
- final claim verdict or review judgment
- optional reviewer-style explanation or critique

## System Modules

### Environment

Responsibilities:
- exposes benchmark sample as initial state
- executes search / retrieval / graph update actions
- records observation after each action
- writes trajectory logs with step-level metadata

Environment should be deterministic where possible so offline replay and evaluation are stable.

### Verifier / Reranker

Responsibilities:
- score evidence relevance
- score support vs contradiction consistency
- score sufficiency of current evidence set
- rerank retrieved candidates or policy proposals

Verifier is trained before heavy policy work because it becomes reusable supervision and evaluation infrastructure.

### Next-action policy

Input:
- serialized state
- candidate action space
- optional verifier summary

Output:
- structured next action JSON

The policy should prioritize evidence gain, contradiction coverage, and graph quality, not direct fluent text generation.

### Stop policy

Input:
- serialized state
- verifier summary
- budget state

Output:
- `should_stop`, `reason`

Stop policy should be trained separately from the next-action policy, then evaluated jointly.

### Reviewer head

Optional head that turns final evidence graph into:
- claim verdict
- review judgment
- critique summary

This head is downstream of evidence seeking and should not be the first training target.

### Benchmark / Eval loop

The benchmark loop should:
- reset environment per sample
- run agent until stop or budget limit
- log full trajectory
- score retrieval, action, stopping, and final task metrics

## Dataset Roles

### SciFact

Use for:
- support / contradict evidence retrieval
- evidence sentence grounding
- verifier supervision
- stop-policy supervision on evidence sufficiency

### FEVER / HoVer

Use for:
- multi-hop fact verification
- contradiction search
- claim decomposition and evidence chaining
- robust stopping under noisy retrieval

### QASPER

Use for:
- document-grounded reasoning over long papers
- span selection and evidence citation
- reviewer-head training for scientific QA style synthesis

### HotpotQA / MuSiQue

Use for:
- next-action policy on multi-hop retrieval
- question decomposition and follow-up actions
- measuring over-search vs correct stop under multi-hop settings

### ReviewBench

Use for:
- review judgment outputs
- critique-style evidence aggregation
- reviewer head and integrated loop evaluation

### DeepReview-13K

Use for:
- review-specific evidence seeking trajectories when aligned with papers / claims
- final review generation conditioned on evidence graph
- offline trajectory mining if annotations allow step extraction

### ReviewCritique

Use for:
- critique generation and rubric-conditioned reviewer head
- verifier supervision for criticism groundedness if labels exist

### AAAR-1.0

Use for:
- review assessment and argument quality
- benchmark-level evaluation of final outputs and evidence-grounded critique

## Training Order

### 1. Trajectory export

First stabilize the environment log format and export scripts. Every later stage depends on consistent trajectories.

### 2. Verifier

Train a lightweight verifier / reranker first using evidence labels and contradiction labels where available.

### 3. Action policy

Train next-action policy with supervised trajectories. Start with offline SFT or LoRA only.

### 4. Stop policy

Train separate stop policy using sufficiency labels or final step annotations.

### 5. Integrated loop

Run policy + verifier + stop policy together in offline replay and benchmark loops.

### 6. Optional RL later

Only after stable offline metrics and reliable trajectory logging:
- reward-model training
- offline RL or DPO-style preference tuning
- online RL in bounded environments if needed

## Experimental Design

### Baselines

Minimum baselines:
- direct reviewer LM without explicit evidence loop
- single-policy SFT with `stop` as normal action
- oracle retrieval + learned stop policy
- learned retrieval / action policy without verifier reranking

### Ablations

Required ablations:
- no contradiction search
- no evidence graph updates
- no verifier features
- unified stop + action head vs separate stop head
- single-hop only vs multi-hop action space

### Metrics

Core metrics:
- evidence recall / precision
- support vs contradiction balance
- next-action accuracy or exact match over structured action JSON
- stop F1 or accuracy
- steps to completion / budget efficiency
- final verdict accuracy or review score
- graph coverage and contradiction resolution metrics

### Benchmark split

Recommended split by function:
- verifier pretraining: SciFact, FEVER, HoVer
- action policy pretraining: HotpotQA, MuSiQue, QASPER
- review-head and integrated evaluation: ReviewBench, DeepReview-13K, ReviewCritique, AAAR-1.0

Keep held-out domains for final integrated evaluation so retrieval and stopping generalization are measurable.

## Engineering Constraints

The system must target:
- single node, `4 x V100-SXM2-32GB`
- `fp16` only
- no `bf16`
- no `flash-attn2`
- no `deepspeed`
- `torchrun` / native DDP only

Implications:
- prefer LoRA or other PEFT for early training
- keep sequence length conservative at first
- use eager attention path
- separate verifier from large policy when possible
- favor modular checkpoints over monolithic joint models

## Recommended Directory Structure

```text
train_agent/
  data/
    adapters/
    manifests/
    processed/
  trajectories/
    schema.py
    recorder.py
    exporters/
  models/
    verifier.py
    action_policy.py
    stop_policy.py
    reviewer_head.py
    prompts.py
  trainers/
    train_verifier.py
    train_action_policy.py
    train_stop_policy.py
    common.py
  eval/
    benchmark_loop.py
    metrics.py
    replay.py
  scripts/
    export_scifact.py
    export_fever_hover.py
    export_qasper.py
    export_reviewbench.py
  configs/
    verifier/
    action/
    stop/
    eval/
```

Migration from the current scaffold:
- current `deep_research_review_v2/trajectory.py` becomes `train_agent/trajectories/schema.py` and `recorder.py`
- current `deep_research_review_v2/export_sft_data.py` becomes one exporter stage under `train_agent/trajectories/exporters/`
- current `deep_research_review_v2/train_sft.py` should be split into task-specific trainers
- current `deep_research_review_v2/verifier.py` is only a placeholder and should evolve into a separate verifier model module

## Minimal First Milestone

A valid first milestone is not full agent training. It is:
- stable trajectory schema
- one exporter from trajectories to `next_action` and `stopping` SFT JSONL
- one verifier baseline
- one action policy LoRA run
- one stop policy LoRA run
- one offline replay evaluation script

That is the minimum stack that supports later integrated training.
