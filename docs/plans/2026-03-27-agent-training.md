# Agent Training Implementation Plan

## Goal

Create the smallest executable path from the current scaffold to a trainable evidence-seeking agent framework for `deep_research_review_v2`, while keeping the work modular enough to support verifier, next-action, stop-policy, and integrated evaluation stages.

## Phase 0: Freeze the current baseline and align structure

### Task 0.1: Move from ad hoc scaffold to planned directory layout

Files to modify:
- `README.md`

Files to create:
- `train_agent/__init__.py`
- `train_agent/data/__init__.py`
- `train_agent/trajectories/__init__.py`
- `train_agent/models/__init__.py`
- `train_agent/trainers/__init__.py`
- `train_agent/eval/__init__.py`
- `train_agent/scripts/__init__.py`
- `train_agent/configs/README.md`

Validation:
- list the package tree in the container
- run compile check for `train_agent`

Minimal test/script first:
- package import smoke test for `train_agent`

Notes:
- do not remove existing `deep_research_review_v2` files yet
- first make the target structure explicit and keep wrappers if needed

### Task 0.2: Record the current trajectory schema as the stable contract

Files to modify:
- `deep_research_review_v2/trajectory.py`

Files to create:
- `train_agent/trajectories/schema.py`
- `train_agent/trajectories/recorder.py`
- `docs/plans/trajectory-schema-notes.md`

Validation:
- round-trip load / save test on a tiny JSONL sample
- schema smoke test under container Python

Minimal test/script first:
- one tiny trajectory round-trip script

## Phase 1: Data export and trajectory logging

### Task 1.1: Separate generic trajectory export from task-specific exporters

Files to modify:
- `deep_research_review_v2/export_sft_data.py`

Files to create:
- `train_agent/trajectories/exporters/base.py`
- `train_agent/trajectories/exporters/next_action.py`
- `train_agent/trajectories/exporters/stopping.py`
- `train_agent/trajectories/exporters/reviewer.py`

Validation:
- export JSONL from demo trajectories
- inspect counts for train / eval split
- inspect one sample per task

Minimal test/script first:
- `train_agent/scripts/export_demo_trajectories.py`

### Task 1.2: Add dataset adapters for public benchmarks

Files to create:
- `train_agent/data/adapters/scifact.py`
- `train_agent/data/adapters/fever.py`
- `train_agent/data/adapters/hover.py`
- `train_agent/data/adapters/qasper.py`
- `train_agent/data/adapters/hotpotqa.py`
- `train_agent/data/adapters/musique.py`
- `train_agent/data/adapters/reviewbench.py`
- `train_agent/data/adapters/deepreview13k.py`
- `train_agent/data/adapters/reviewcritique.py`
- `train_agent/data/adapters/aaar.py`
- `train_agent/data/manifests/datasets.yaml`

Validation:
- each adapter exports at least one normalized record format
- manifest dry run reports sample counts and missing fields

Minimal test/script first:
- adapter check script starting with `scifact`

Notes:
- start with `SciFact`, `FEVER/HoVer`, and one multi-hop dataset first
- review datasets follow after the generic pipeline is stable

### Task 1.3: Add trajectory replay generation

Files to create:
- `train_agent/trajectories/replay.py`
- `train_agent/scripts/generate_trajectories.py`

Validation:
- deterministic replay over demo inputs
- trajectory JSONL contains state, action, observation, verifier summary, and stop label

Minimal test/script first:
- generate 5 toy trajectories from local examples

## Phase 2: Verifier and reranker

### Task 2.1: Replace heuristic verifier with trainable baseline

Files to modify:
- `deep_research_review_v2/verifier.py`

Files to create:
- `train_agent/models/verifier.py`
- `train_agent/trainers/train_verifier.py`
- `train_agent/configs/verifier/base.yaml`

Validation:
- train on a tiny split in the container conda env
- report validation accuracy or F1 on support vs contradiction labels

Minimal test/script first:
- one-batch forward pass script

### Task 2.2: Add evidence reranking interface

Files to create:
- `train_agent/models/reranker.py`
- `train_agent/eval/rerank_eval.py`

Validation:
- candidate list in, reranked candidate list out
- measure top-k relevance lift over raw retrieval order

Minimal test/script first:
- feed 3 mock candidates and verify stable ranking output format

## Phase 3: Next-action policy

### Task 3.1: Split generic SFT trainer into reusable common utilities

Files to modify:
- `deep_research_review_v2/train_sft.py`

Files to create:
- `train_agent/trainers/common.py`
- `train_agent/trainers/train_action_policy.py`
- `train_agent/models/action_policy.py`
- `train_agent/configs/action/base.yaml`

Validation:
- 4-GPU torchrun launch works under fp16
- one mini epoch on toy data finishes
- output checkpoint and tokenizer are saved

Minimal test/script first:
- single-process overfit on 8 examples before DDP

Notes:
- keep LoRA only
- force `bf16=False`
- use eager attention path

### Task 3.2: Add structured action decoding checks

Files to create:
- `train_agent/eval/action_metrics.py`
- `train_agent/scripts/check_action_json.py`

Validation:
- generated actions parse as JSON
- exact match and field-level accuracy are reported

Minimal test/script first:
- check 10 exported labels and 10 generated samples

## Phase 4: Stop policy

### Task 4.1: Create dedicated stop-policy model and trainer

Files to create:
- `train_agent/models/stop_policy.py`
- `train_agent/trainers/train_stop_policy.py`
- `train_agent/configs/stop/base.yaml`

Validation:
- train and eval on stopping JSONL
- report stop accuracy, stop F1, and average extra steps under replay

Minimal test/script first:
- single-batch training smoke test

### Task 4.2: Add sufficiency features from verifier

Files to modify:
- `train_agent/models/stop_policy.py`
- `train_agent/trajectories/replay.py`

Validation:
- stop policy accepts verifier-derived features
- ablation with and without verifier features

Minimal test/script first:
- replay one trajectory and inspect stop input payload

## Phase 5: Integrated loop

### Task 5.1: Build benchmark replay loop

Files to create:
- `train_agent/eval/benchmark_loop.py`
- `train_agent/eval/metrics.py`
- `train_agent/configs/eval/base.yaml`

Validation:
- run action policy plus verifier plus stop policy offline on a tiny held-out set
- log trajectory, final evidence graph summary, and benchmark metrics

Minimal test/script first:
- replay 3 samples end to end without distributed training

### Task 5.2: Add reviewer head placeholder

Files to create:
- `train_agent/models/reviewer_head.py`
- `train_agent/trainers/train_reviewer_head.py`

Validation:
- consume final evidence graph and produce structured summary or critique
- evaluate on one review-style dev subset

Minimal test/script first:
- format-only generation from gold evidence graph

## Phase 6: Optional RL later

### Task 6.1: Define reward model inputs and offline reward logs

Files to create:
- `train_agent/models/reward_model.py`
- `train_agent/scripts/export_reward_data.py`

Validation:
- reward examples can be exported from trajectories
- reward targets include retrieval gain, contradiction coverage, and stop correctness

Minimal test/script first:
- export reward rows from 10 trajectories

### Task 6.2: Evaluate whether RL is justified

Validation questions:
- is action accuracy saturated under SFT?
- is stop quality still poor after verifier features?
- does integrated replay show strong exposure bias?

Only proceed to RL if offline evidence says SFT plus verifier plus stop policy is insufficient.

## Recommended Execution Order

1. stabilize directory structure and schema
2. modularize trajectory export
3. implement 2 to 3 dataset adapters
4. train first verifier baseline
5. train next-action policy
6. train stop policy
7. run integrated offline benchmark loop
8. add reviewer head only after the loop works
9. consider reward modeling and RL later

## Verification Checklist Per Milestone

For every milestone, verify through container commands only:
- compile check in the container
- minimal smoke script for shape and schema correctness
- tiny local dataset run before multi-GPU launch
- 4-GPU torchrun only after single-process success
- logs written under the repository for later inspection

## Minimal Runnable Version Definition

A milestone counts as complete only if it delivers all of the following:
- trajectory logging works on toy inputs
- exporter writes `next_action` and `stopping` datasets
- verifier baseline trains on one public dataset
- action policy trains on one toy or small real split
- stop policy trains on the same replay format
- integrated replay reports action and stopping metrics

Anything beyond that is optional until the loop is stable.
