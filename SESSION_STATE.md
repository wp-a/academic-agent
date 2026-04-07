# Session State

## Current Scope

This repository is currently focused on the **modular evidence-seeking agent** mainline:

- frozen verifier
- hard replay / off-policy export
- action policy / stop policy imitation
- stronger teacher relabel
- DAgger-style mixed-train loop

Do **not** branch into full reviewer head, online RL, PPO, or verifier retraining unless explicitly requested.

## Environment

- Host repo path: `/public/localUsers/wangpengv2/AcademicSubmission`
- Container repo path: `/mnt/AcademicSubmission`
- Container: `2e230dfba7c5`
- All Python / data / train / eval commands must run via:
  `docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && ...'`
- Preferred Python inside container:
  `/root/miniconda3/bin/python`
- Hardware assumption:
  single node `4 x V100-SXM2-32GB`
- Precision / infra constraints:
  `fp16` only, no `bf16`, no `flash-attn2`, no `deepspeed`

## Proxy

When container commands need external access, use:

```sh
export HTTP_PROXY=http://172.17.0.1:17898
export HTTPS_PROXY=http://172.17.0.1:17898
export NO_PROXY=localhost,127.0.0.1
```

## Current Mainline Status

### Verifier

- frozen verifier mainline:
  `outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin`
- baseline:
  `v7_pairwise_margin + full_document`
- this remains frozen for current work

### Action Policy

- task:
  `search / quote_evidence / stop`
- state text source:
  `train_agent/rl/restricted_retrieval.py::RestrictedRetrievalState.to_text()`
- hard mainline model:
  `outputs/action_policy_scifact_hard_qwen25_3b_lora_v1`
- hard validation step metrics:
  - `accuracy = 0.997845`
  - `macro_f1 = 0.997776`

### Stop Policy

- task:
  `no / yes`
- hard mainline model:
  `outputs/stop_policy_scifact_hard_qwen25_3b_lora_v1`
- hard validation step metrics:
  - `accuracy = 0.998563`
  - `macro_f1 = 0.997962`

### Hard Joint Offline Replay

- output:
  `outputs/joint_policy_scifact_hard_qwen25_3b_lora_v1/offline_replay_validation_hard.json`
- teacher-aligned metrics:
  - `action_agreement = 0.998564`
  - `stop_recall = 0.996865`
  - `success_rate = 0.97929`
  - `mismatch_episode_count = 2`
  - `mismatch_step_count = 2`
  - `off_policy_episode_count = 2`
  - `off_policy_action_examples = 3`
  - `off_policy_stop_examples = 3`

Interpretation:

- hard replay export and eval reference are now aligned on `ConservativeReplayPolicy`
- the old low hard replay numbers were historical reference mismatch, not the current mainline conclusion

## Current Failure Analysis State

### Mismatch Analysis

Script:
- `train_agent/scripts/analyze_hard_replay_mismatches.py`

Current hard validation buckets:
- `premature_stop_after_evidence = 1`
- `oversearch_after_quote = 1`

Current hard validation error-source interpretation:
- `stop_policy_false_positive = 1`
- `stop_policy_false_negative = 1`

### Residual Off-Policy Artifacts

- diagnostics:
  `outputs/joint_policy_scifact_hard_qwen25_3b_lora_v1/offline_replay_validation_hard_mismatch_episodes.jsonl`
- off-policy action:
  `outputs/joint_policy_scifact_hard_qwen25_3b_lora_v1/off_policy_action_validation_hard.jsonl`
- off-policy stop:
  `outputs/joint_policy_scifact_hard_qwen25_3b_lora_v1/off_policy_stop_validation_hard.jsonl`

Semantics:

- once student first deviates from the reference teacher, export that step and all later student-visited states as training-ready JSONL

## DAgger / Stronger Teacher State

### Key Scripts

- mismatch analysis:
  `train_agent/scripts/analyze_hard_replay_mismatches.py`
- stronger teacher relabel:
  `train_agent/scripts/build_stronger_teacher_relabels.py`
- relabel merge helper:
  `train_agent/scripts/merge_relabel_into_trainset.py`
- mixed trainset builder:
  `train_agent/scripts/build_mixed_trainset.py`
- minimal DAgger driver:
  `train_agent/scripts/run_minimal_dagger_round.py`

### Stronger Teacher Backend

`build_stronger_teacher_relabels.py` now supports:

- `teacher_backend = rule_based`
- `teacher_backend = llm_api`

Important invariants:

- keep existing training-compatible action / stop JSONL schema
- preserve `decision_type`
- preserve `uncertain_skip`
- preserve `mixed train` downstream interface

Current `llm_api` backend status:

- interface is implemented
- tests pass
- real network-backed annotation has **not** yet been validated in production
- backend failure should route records to `uncertain_skip`, not silently poison training

### Relabel Metadata

Current relabel records carry:

- `metadata.failure_bucket`
- `metadata.teacher_backend`
- `metadata.teacher_type`
- `metadata.teacher_version`
- `metadata.teacher_label_action`
- `metadata.teacher_label_stop`
- `metadata.teacher_confidence`
- `metadata.relabel_decision_type`

Current relabel summary includes:

- `decision_type_distribution`
- `decision_type_episode_ids`
- `uncertain_skip_action_records`
- `uncertain_skip_stop_records`

### Mixed Train

`build_mixed_trainset.py` produces trainer-compatible files that keep only the six required training fields:

- `trajectory_id`
- `step_id`
- `task`
- `text`
- `label`
- `label_text`

Do not reintroduce raw `metadata` into mixed train outputs; that previously broke the current trainer schema.

### Minimal DAgger Driver

`run_minimal_dagger_round.py` now supports:

- full mini round:
  off-policy export -> relabel -> mixed train -> tiny smoke compare
- explicit preset:
  `--preset export_relabel_mix_only`

That preset should:

- run export / relabel / mix
- skip training and replay compare
- be the preferred mode when scaling off-policy collection

## Last Known DAgger Results

### Leakage-Safe Train-Split Off-Policy

Real train-split hard export produced:

- `episodes = 957`
- `mismatch_episode_count = 1`
- `off_policy_action_examples = 2`
- `off_policy_stop_examples = 2`

This is the current bottleneck:

- train-split mismatch volume is too small
- next leverage should come from collecting more off-policy states, not from immediately scaling the student model

### Rule-Based Relabel + Mixed Train

Output dir:
- `outputs/dagger_scifact_hard_train_rule_relabel_v1`

Summary:
- `episodes_relabeled = 1`
- `action_records_relabeled = 2`
- `stop_records_relabeled = 2`
- bucket:
  `oversearch_after_quote`

### Tiny Smoke Compare

Action step-level:
- base:
  `accuracy = 0.961925`
- mixed:
  `accuracy = 0.971264`

Stop step-level:
- base:
  `accuracy = 0.915948`
- mixed:
  `accuracy = 0.932471`

Hard validation joint replay:
- base:
  - `action_agreement = 0.895710`
  - `stop_recall = 0.818841`
  - `success_rate = 0.878698`
  - `early_stop_rate = 0.103550`
- mixed:
  - `action_agreement = 0.907028`
  - `stop_recall = 0.845070`
  - `success_rate = 0.890533`
  - `early_stop_rate = 0.091716`

Interpretation:

- even a tiny leakage-safe DAgger update gave a positive directional signal
- but the available off-policy train data is still far too small for strong conclusions

## What To Do Next After Restart

Do these in order:

1. keep the frozen verifier unchanged
2. use `run_minimal_dagger_round.py --preset export_relabel_mix_only` to expand off-policy collection
3. run a small real `teacher_backend=llm_api` relabel batch
4. inspect:
   - `decision_type_distribution`
   - `decision_type_episode_ids`
   - `uncertain_skip` ratio
5. only then mix accepted relabel data into training and run the next DAgger comparison

Do **not** jump to:

- full reviewer head
- PPO / online RL
- verifier retraining
- bigger student scaling before off-policy collection expands

## Recovery Prompt

If context is lost, restore with this mental model:

- the project is no longer a demo scaffold; it is a modular evidence-seeking agent pipeline
- the mainline is:
  frozen verifier -> hard replay -> off-policy export -> stronger teacher relabel -> mixed train -> DAgger-style compare
- current highest-value next step is:
  expand off-policy collection, then validate real LLM stronger teacher labels
- current `teacher_backend=llm_api` interface is implemented, but still needs real annotation validation
- current goal is still **not** full reviewer generation; it is to strengthen the evidence-seeking control loop first
