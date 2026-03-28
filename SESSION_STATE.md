# Session State

## Current Scope

This repository is currently focused on the **action policy** training loop only.
Do not branch into stop policy, verifier, or RL until explicitly requested.

## Repository

- Project root: `/mnt/AcademicSubmission`
- Container: `2e230dfba7c5`
- All Python / training / data commands should run through:
  `docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && ...'`
- Conda env inside container:
  `/root/miniconda3/bin/conda run -n base ...`

## Proxy Workaround

The container does not natively inherit host proxy settings.
Current working workaround:

1. Host has local proxy at `127.0.0.1:17897`
2. A host-side TCP forwarder was started to expose it as `0.0.0.0:17898`
3. Container uses host gateway address `172.17.0.1:17898`

Use these env vars in container commands when external access is needed:

```sh
export HTTP_PROXY=http://172.17.0.1:17898
export HTTPS_PROXY=http://172.17.0.1:17898
export NO_PROXY=localhost,127.0.0.1
```

Note: if host or session restarts, the forwarder may need to be started again.

## Installed / Prepared Extras

### Superpowers

Installed locally:
- repo: `/public/localUsers/wangpengv2/.codex/superpowers`
- symlink: `/public/localUsers/wangpengv2/.agents/skills/superpowers`

Codex may need restart to pick it up.

### agency-agents

Inspected at `/tmp/agency-agents`.
It is **not** a Codex skill pack; it is a multi-tool agent library.
Not yet wired into Codex.

## Action Policy Status

### Current framing

The action policy is now treated as a **classification task**, not JSON generation.

- Input: `state_text`
  - contains claim, hypothesis, observation, history, evidence, action space
- Output: discrete `action_type`
- Labels currently observed:
  - `ask_followup`
  - `quote_evidence`
  - `search`
  - `stop`
  - `update_graph`
- Loss: cross-entropy via sequence classification head

### Key files

- Exporter:
  `/public/localUsers/wangpengv2/AcademicSubmission/train_agent/trajectories/export_from_deep_review.py`
- Split script:
  `/public/localUsers/wangpengv2/AcademicSubmission/train_agent/scripts/split_action_data.py`
- Trainer:
  `/public/localUsers/wangpengv2/AcademicSubmission/train_agent/trainers/train_action_policy.py`
- Common trainer utils:
  `/public/localUsers/wangpengv2/AcademicSubmission/train_agent/trainers/common.py`
- Inference script:
  `/public/localUsers/wangpengv2/AcademicSubmission/train_agent/scripts/infer_action_policy.py`
- Helper to write one eval example to text:
  `/public/localUsers/wangpengv2/AcademicSubmission/train_agent/scripts/write_first_eval_text.py`
- One-shot smoke pipeline:
  `/public/localUsers/wangpengv2/AcademicSubmission/scripts/run_action_policy_smoke.sh`

## Model / Outputs

### Downloaded small pretrained classifier model

- `prajjwal1/bert-tiny`
- cached at:
  `/public/localUsers/wangpengv2/AcademicSubmission/models/hf_cache/prajjwal1_bert_tiny`

### Main training output already completed

- output dir:
  `/public/localUsers/wangpengv2/AcademicSubmission/outputs/action_policy_bert_tiny`
- important files:
  - `model.safetensors`
  - `label_names.json`
  - `train_metrics.json`
  - `training_args.json`
  - `checkpoint-2/`
  - `checkpoint-4/`

### Smoke pipeline output

- output dir:
  `/public/localUsers/wangpengv2/AcademicSubmission/outputs/action_policy_smoke_pipeline`

## Last Known Results

### Real tiny-BERT action-policy train

Approx final metrics from completed run:
- train loss: `1.5142`
- final eval loss: `1.7121`

### Latest smoke pipeline run

Approx final metrics:
- train loss: `1.5645`
- final eval loss: `1.5540`

### Latest inference result

Inference on one held-out state text predicted:
- `predicted_label_id = 2`
- `predicted_action_type = search`

## Public Dataset Status

SciFact access works through proxy, but download command still needs correct config name.
The dataset requires explicit config such as:
- `corpus`
- `claims`

It also requires `trust_remote_code=True`.

## Recommended Next Step After Restart

Resume with **action policy only**.
Suggested next task:
1. download SciFact `claims`
2. inspect schema
3. map it into `state_text -> action_type` or related action-policy supervision
4. add eval accuracy script for current classifier outputs

## Reusable Commands

### Export action classification data

```sh
docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && /root/miniconda3/bin/conda run -n base python -m train_agent.trajectories.export_from_deep_review --input data/demo_trajectories.jsonl --output data/exports/train_agent_action_classification.jsonl'
```

### Split train / eval

```sh
docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && /root/miniconda3/bin/conda run -n base python -m train_agent.scripts.split_action_data --input data/exports/train_agent_action_classification.jsonl --train_output data/exports/train_agent_action_cls_train.jsonl --eval_output data/exports/train_agent_action_cls_eval.jsonl --eval_ratio 0.25 --seed 42'
```

### Train action policy

```sh
docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && /root/miniconda3/bin/conda run -n base python -m train_agent.trainers.train_action_policy --train_file data/exports/train_agent_action_cls_train.jsonl --eval_file data/exports/train_agent_action_cls_eval.jsonl --model_name_or_path models/hf_cache/prajjwal1_bert_tiny --output_dir outputs/action_policy_bert_tiny --max_length 256 --max_steps 4 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --logging_steps 1 --eval_steps 1 --save_steps 2'
```

### Run inference

```sh
docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && /root/miniconda3/bin/conda run -n base python -m train_agent.scripts.infer_action_policy --model_dir outputs/action_policy_bert_tiny --state_text_file outputs/tmp/infer_eval_text.txt --max_length 256'
```

### Run full smoke pipeline

```sh
docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && bash scripts/run_action_policy_smoke.sh'
```
