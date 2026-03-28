# Session State

## Current Scope

This repository is currently focused on the **action policy + stop policy** training loop.
Verifier remains frozen. Do not branch into joint verifier retraining or RL until explicitly requested.

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
- Step-level eval script:
  `/public/localUsers/wangpengv2/AcademicSubmission/train_agent/scripts/eval_action_policy_predictions.py`
  - now supports `prediction_rows` / `error_rows` export
  - checkpoint directories can inherit `label_names.json` from the parent output dir
- Helper to write one eval example to text:
  `/public/localUsers/wangpengv2/AcademicSubmission/train_agent/scripts/write_first_eval_text.py`
- One-shot smoke pipeline:
  `/public/localUsers/wangpengv2/AcademicSubmission/scripts/run_action_policy_smoke.sh`

## Stop Policy Status

### Current framing

The stop policy is now treated as a **classification task** over the same verifier-driven `state_text`.

- Input: `state_text`
- Output: discrete `should_stop`
- Labels currently used:
  - `no`
  - `yes`

### Key files

- SciFact stop replay exporter:
  `/public/localUsers/wangpengv2/AcademicSubmission/train_agent/scripts/export_scifact_stop_policy_data.py`
- Trainer currently reused for stop classification:
  `/public/localUsers/wangpengv2/AcademicSubmission/train_agent/trainers/train_action_policy.py`

### Current data

- output dir:
  `/public/localUsers/wangpengv2/AcademicSubmission/data/processed/scifact_stop_policy_v1`
- train examples: `2974`
- validation examples: `1046`
- class balance:
  - train `no=2059`, `yes=915`
  - validation `no=724`, `yes=322`

### Current stop-policy output

- output dir:
  `/public/localUsers/wangpengv2/AcademicSubmission/outputs/stop_policy_scifact_qwen25_3b_lora_v1`

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

### Latest step-level eval result

Real SciFact validation eval via `eval_action_policy_predictions` produced:
- `num_examples = 1046`
- `accuracy = 0.997132`
- `macro_f1 = 0.997020`
- `num_errors = 3`
- error file:
  `/public/localUsers/wangpengv2/AcademicSubmission/outputs/action_policy_scifact_bert_tiny_v1/step_eval_validation_errors.json`

### Latest checkpoint comparison

Direct eval now works for:
- `outputs/action_policy_scifact_bert_tiny_v1/checkpoint-920`
- `outputs/action_policy_scifact_bert_tiny_v1/checkpoint-930`

Observed result:
- `checkpoint-920`, `checkpoint-930`, and final output dir all matched on current step-level metrics
- current residual errors are three `step_id=1` cases
- two mistakes are `quote_evidence -> stop`
- one mistake is `quote_evidence -> search`

### Latest stop-policy train result

Four-GPU `Qwen2.5-3B-Instruct + LoRA` stop training completed on SciFact stop replay.

Final validation metrics:
- `accuracy = 1.0`
- `macro_f1 = 1.0`
- confusion matrix:
  - `[[724, 0], [0, 322]]`

### Latest joint replay result

Joint replay with:
- `outputs/action_policy_scifact_qwen25_3b_lora_v1`
- `outputs/stop_policy_scifact_qwen25_3b_lora_v1`
- frozen verifier `v7_pairwise_margin + full_document`

Produced:
- `action_agreement = 1.0`
- `success_rate = 0.973373`
- `early_stop_rate = 0.0`
- `suppressed_stop_count = 0`

Interpretation:
- joint replay matched the existing action-only Qwen replay on current SciFact weak labels
- current bottleneck is no longer epoch count on this dataset
- next leverage should come from larger and harder replay data, not more training passes over the same weak labels

## Public Dataset Status

SciFact access works through proxy, but download command still needs correct config name.
The dataset requires explicit config such as:
- `corpus`
- `claims`

It also requires `trust_remote_code=True`.

## Recommended Next Step After Restart

Resume with **action policy + stop policy**.
Suggested next task:
1. export larger and harder replay data beyond current SciFact weak labels
2. prioritize multi-hop and longer-horizon supervision for `action policy` and `stop policy`
3. keep the verifier frozen while scaling data and rerun joint replay
4. only increase epochs after new data stops improving joint replay

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

### Run step-level action-policy eval

```sh
docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && /root/miniconda3/bin/conda run -n base python -m train_agent.scripts.eval_action_policy_predictions --model_dir outputs/action_policy_scifact_bert_tiny_v1 --eval_file data/processed/scifact_action_policy_v1/scifact_action_policy_validation.jsonl --output_path outputs/action_policy_scifact_bert_tiny_v1/step_eval_validation.json --errors_output_path outputs/action_policy_scifact_bert_tiny_v1/step_eval_validation_errors.json --max_length 256 --batch_size 32'
```

### Export SciFact stop-policy replay data

```sh
docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && export HTTP_PROXY=http://172.17.0.1:17898 && export HTTPS_PROXY=http://172.17.0.1:17898 && export NO_PROXY=localhost,127.0.0.1 && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.scripts.export_scifact_stop_policy_data --verifier_model_name_or_path outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin --output_dir data/processed/scifact_stop_policy_v1 --max_steps 4 --doc_aggregation full_document --aggregation_top_k 3 --max_length 384 --batch_size 8 --trust_remote_code'
```

### Train stop policy on 4 GPUs

```sh
docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && export CUDA_VISIBLE_DEVICES=0,1,2,3 && /root/miniconda3/bin/torchrun --nproc_per_node=4 -m train_agent.trainers.train_action_policy --train_file data/processed/scifact_stop_policy_v1/scifact_stop_policy_train.jsonl --eval_file data/processed/scifact_stop_policy_v1/scifact_stop_policy_validation.jsonl --model_name_or_path /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 --output_dir outputs/stop_policy_scifact_qwen25_3b_lora_v1 --max_length 512 --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 1e-4 --logging_steps 10 --eval_steps 50 --save_steps 50 --attn_implementation sdpa --use_lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --lora_target_modules q_proj,k_proj,v_proj,o_proj --lora_modules_to_save score --gradient_checkpointing'
```

### Run joint action+stop offline replay

```sh
docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.scripts.eval_action_policy_offline_replay --policy_model_dir outputs/action_policy_scifact_qwen25_3b_lora_v1 --stop_model_dir outputs/stop_policy_scifact_qwen25_3b_lora_v1 --verifier_model_name_or_path outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin --output_path outputs/joint_policy_scifact_qwen25_3b_lora_v1/offline_replay_validation.json --split validation --max_steps 4 --policy_max_length 512 --policy_batch_size 8 --stop_max_length 512 --stop_batch_size 8 --verifier_max_length 384 --verifier_batch_size 8 --doc_aggregation full_document --aggregation_top_k 3'
```

### Run full smoke pipeline

```sh
docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && bash scripts/run_action_policy_smoke.sh'
```
