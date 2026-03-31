# Off-Policy State Collection Design

## Goal

Add a minimal, training-ready export path on top of `train_agent/scripts/eval_action_policy_offline_replay.py` that captures student-visited off-policy states during offline replay.

The export is intended to support the next stronger-teacher and DAgger stage without introducing a parallel data pipeline.

## Scope

This design only changes offline replay export behavior.

It does not:
- change the current scoring metrics
- change action or stop prediction behavior
- introduce LLM labeling yet
- retrain any model yet
- change the existing diagnostics JSONL contract

## Problem

The current evaluator can now export mismatch diagnostics, but those records are optimized for inspection, not for direct reuse in the current action / stop training loop.

For DAgger-style augmentation, we need records that:
- come from the states actually visited by the student policy
- preserve the current `text` field generated from `RestrictedRetrievalState.to_text()`
- use teacher labels in the same shape as existing action / stop training JSONL
- can be mixed into the current training pipeline without an extra converter

## Chosen Approach

Use the current offline replay loop to export `student-visited off-policy states`.

Definition:
- For each episode, once the student action first differs from the reference teacher action, mark the episode as off-policy.
- Export the current step and every subsequent step along the actual student-visited trajectory.
- Label each exported state with the current reference teacher action for that state.

This produces a DAgger-style replay slice while staying inside the current SciFact hard replay evaluator.

## Why This Approach

Compared with exporting only mismatch steps, this keeps the trajectory after divergence, which is the real compounding-error regime we care about.

Compared with exporting only diagnostics JSONL and post-processing later, this keeps the data immediately usable by the current action and stop trainers.

## Output Files

Keep the existing optional diagnostics output:
- `--diagnostics_output_path`

Add two new optional outputs:
- `--off_policy_action_output_path`
- `--off_policy_stop_output_path`

These paths are independent. Users may request only action export, only stop export, both, or neither.

## Action Export Format

Each off-policy action record must reuse the existing action-policy training shape:
- `trajectory_id`
- `step_id`
- `task`
- `text`
- `label`
- `label_text`

Field definitions:
- `trajectory_id`: episode id
- `step_id`: environment step index for the exported state
- `task`: `next_action_classification`
- `text`: current `state.to_text()` from the student-visited state
- `label`: reference teacher action for that state
- `label_text`: `{"action_type": ...}`

Add a `metadata` field for provenance and filtering:
- `episode_id`
- `student_action`
- `reference_action`
- `is_first_off_policy_step`
- `reference_policy_type`
- `post_quote_search_budget`
- `used_stop_policy`
- `stop_policy_should_stop`
- `suppressed_stop`

The metadata is additive and should not change current trainer compatibility for code paths that ignore metadata.

## Stop Export Format

Stop export should reuse the existing stop-policy conversion logic by passing the off-policy action-style record through `convert_action_record_to_stop_record()`.

This preserves the current stop training shape:
- `trajectory_id`
- `step_id`
- `task`
- `text`
- `label`
- `label_text`

The resulting stop label should still be:
- `yes` when teacher action is `stop`
- `no` otherwise

The `metadata` block from the off-policy action record should be preserved when feasible so later filtering can distinguish true off-policy contexts.

## Export Semantics

Episode-level semantics:
- If an episode never diverges from the teacher, export nothing to off-policy action / stop outputs.
- If an episode diverges, count it once in `off_policy_episode_count`.

Step-level semantics:
- The first mismatched step is exported.
- Every later student-visited step in the same episode is exported, even if the student re-aligns with the teacher later.
- Labels are always recomputed from the teacher on the current student-visited state.

## Summary Fields

Extend the evaluator summary with:
- `off_policy_episode_count`
- `off_policy_action_examples`
- `off_policy_stop_examples`
- `off_policy_action_output_path`
- `off_policy_stop_output_path`

These fields should default to zero / empty string when no export is requested.

## CLI Changes

Add optional CLI arguments to `train_agent/scripts/eval_action_policy_offline_replay.py`:
- `--off_policy_action_output_path`
- `--off_policy_stop_output_path`

The evaluator must continue to work unchanged when neither argument is provided.

## Testing

Add test coverage in `tests/test_action_policy_offline_eval.py` for:
- exporting off-policy action records only after the first divergence
- exporting the first mismatch step plus subsequent student-visited states
- writing teacher labels in current training-ready action format
- writing stop records in current stop-policy format
- updating summary counts and output paths correctly

Tests should continue to cover the existing diagnostics export.

## Non-Goals For This Step

This step does not yet:
- call an LLM teacher
- merge exported off-policy records into training data
- change `train_action_policy.py` or stop training scripts
- add dataset mixing logic
- change replay metrics or success criteria

## Implementation Notes

The implementation should stay localized to:
- `train_agent/scripts/eval_action_policy_offline_replay.py`
- `tests/test_action_policy_offline_eval.py`

If the current evaluator needs small helper functions for record serialization, keep them in the same script.

## Verification

Minimum verification for this step:
- `docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && /root/miniconda3/bin/python -m pytest tests/test_action_policy_offline_eval.py tests/test_scifact_hard_replay.py -v'`

Optional follow-up verification after implementation:
- run hard validation with both new output paths set and inspect the produced JSONL counts
