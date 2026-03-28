# Reproduce

本文档给出当前仓库最关键的一条可复现实验链：

- 冻结 relevance 在 `v7_pairwise_margin + full_document`
- 基于 frozen verifier 导出 SciFact weakly-coupled action replay 数据
- 训练 action policy 小分类 baseline
- 训练 `Qwen2.5-3B-Instruct` LoRA action policy
- 在 restricted env 中做 episode-level offline replay 评测

本文档不覆盖：

- rollout 主循环
- online RL
- verifier 新 loss 试验
- FEVER / HoVer 混训

## 1. 环境

默认路径：

- 宿主机：`/public/localUsers/wangpengv2/AcademicSubmission`
- 容器：`/mnt/AcademicSubmission`
- Docker 容器：`2e230dfba7c5`

默认约束：

- `4 x V100-SXM2-32GB`
- `fp16`
- `torchrun` / DDP
- 不使用 `bf16`
- 不使用 `flash-attn2`
- 不使用 `deepspeed`

所有 Python / 训练 / 数据处理命令都通过容器执行：

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && ..."
```

## 2. 代理

访问 Hugging Face 时需要：

```sh
export HTTP_PROXY=http://172.17.0.1:17898
export HTTPS_PROXY=http://172.17.0.1:17898
export NO_PROXY=localhost,127.0.0.1
```

## 3. Frozen Verifier 基线

当前 frozen verifier 固定为：

- `outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin`
- 文档聚合：`full_document`

这是当前 action replay 导出和 offline replay 评测默认使用的 verifier。

## 4. 导出 SciFact Action Replay 数据

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export HTTP_PROXY=http://172.17.0.1:17898 && export HTTPS_PROXY=http://172.17.0.1:17898 && export NO_PROXY=localhost,127.0.0.1 && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.scripts.export_scifact_action_policy_data --verifier_model_name_or_path outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin --output_dir data/processed/scifact_action_policy_v1 --max_steps 4 --doc_aggregation full_document --aggregation_top_k 3 --max_length 384 --batch_size 8"
```

产物：

- `data/processed/scifact_action_policy_v1/scifact_action_policy_train.jsonl`
- `data/processed/scifact_action_policy_v1/scifact_action_policy_validation.jsonl`
- `data/processed/scifact_action_policy_v1/export_summary.json`

当前数据规模：

- train：`957` episodes，`2974` action examples
- validation：`338` episodes，`1046` action examples

动作集合：

- `quote_evidence`
- `search`
- `stop`

## 5. 训练小分类 Baseline

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export HTTP_PROXY=http://172.17.0.1:17898 && export HTTPS_PROXY=http://172.17.0.1:17898 && export NO_PROXY=localhost,127.0.0.1 && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.trainers.train_action_policy --train_file data/processed/scifact_action_policy_v1/scifact_action_policy_train.jsonl --eval_file data/processed/scifact_action_policy_v1/scifact_action_policy_validation.jsonl --model_name_or_path prajjwal1/bert-tiny --output_dir outputs/action_policy_scifact_bert_tiny_v1 --max_length 256 --num_train_epochs 5 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --learning_rate 2e-4 --logging_steps 20 --eval_steps 40 --save_steps 40 --gradient_accumulation_steps 1 --attn_implementation sdpa"
```

验证集 action-level 指标：

- `accuracy = 0.997132`
- `macro_f1 = 0.99702`
- per-action f1：
  - `quote_evidence = 0.99542`
  - `search = 0.998736`
  - `stop = 0.996904`

## 6. 预下载 Qwen2.5-3B-Instruct

建议先单进程拉取 snapshot：

```sh
docker exec -i 2e230dfba7c5 bash -lc "export HTTP_PROXY=http://172.17.0.1:17898 && export HTTPS_PROXY=http://172.17.0.1:17898 && export NO_PROXY=localhost,127.0.0.1 && export HF_HUB_DOWNLOAD_TIMEOUT=120 && export HF_HUB_ETAG_TIMEOUT=120 && /root/miniconda3/bin/python - <<\"PY\"
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=\"Qwen/Qwen2.5-3B-Instruct\", max_workers=1, allow_patterns=[\"*.json\", \"*.safetensors\", \"tokenizer*\", \"merges.txt\", \"vocab.json\", \"*.model\", \"*.py\"]))
PY"
```

本次实验使用的本地 snapshot：

- `/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1`

## 7. 训练 Qwen2.5-3B-Instruct LoRA Action Policy

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export CUDA_VISIBLE_DEVICES=0,1,2,3 && /root/miniconda3/bin/torchrun --nproc_per_node=4 -m train_agent.trainers.train_action_policy --train_file data/processed/scifact_action_policy_v1/scifact_action_policy_train.jsonl --eval_file data/processed/scifact_action_policy_v1/scifact_action_policy_validation.jsonl --model_name_or_path /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 --output_dir outputs/action_policy_scifact_qwen25_3b_lora_v1 --max_length 512 --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 1e-4 --logging_steps 10 --eval_steps 50 --save_steps 50 --attn_implementation sdpa --use_lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --lora_target_modules q_proj,k_proj,v_proj,o_proj --lora_modules_to_save score --gradient_checkpointing"
```

训练输出：

- `outputs/action_policy_scifact_qwen25_3b_lora_v1`

验证集 action-level 指标：

- `accuracy = 1.0`
- `macro_f1 = 1.0`
- per-action f1：
  - `quote_evidence = 1.0`
  - `search = 1.0`
  - `stop = 1.0`

## 8. Offline Replay 评测

### 8.1 bert-tiny

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.scripts.eval_action_policy_offline_replay --policy_model_dir outputs/action_policy_scifact_bert_tiny_v1 --verifier_model_name_or_path outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin --output_path outputs/action_policy_scifact_bert_tiny_v1/offline_replay_validation.json --split validation --max_steps 4 --policy_max_length 256 --policy_batch_size 32 --verifier_max_length 384 --verifier_batch_size 8 --doc_aggregation full_document --aggregation_top_k 3"
```

结果：

- `action_agreement = 0.995215`
- `average_steps = 3.091716`
- `stop_precision = 0.993769`
- `stop_recall = 1.0`
- `quote_evidence_hit_rate = 1.0`
- `success_rate = 0.964497`
- `early_stop_rate = 0.005917`

### 8.2 Qwen2.5-3B-Instruct LoRA

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.scripts.eval_action_policy_offline_replay --policy_model_dir outputs/action_policy_scifact_qwen25_3b_lora_v1 --verifier_model_name_or_path outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin --output_path outputs/action_policy_scifact_qwen25_3b_lora_v1/offline_replay_validation.json --split validation --max_steps 4 --policy_max_length 512 --policy_batch_size 8 --verifier_max_length 384 --verifier_batch_size 8 --doc_aggregation full_document --aggregation_top_k 3"
```

结果：

- `action_agreement = 1.0`
- `average_steps = 3.094675`
- `stop_precision = 1.0`
- `stop_recall = 1.0`
- `quote_evidence_hit_rate = 1.0`
- `success_rate = 0.973373`
- `early_stop_rate = 0.0`

## 9. 结果解读

当前这组结果说明：

- frozen verifier 到 action policy 的接口已经稳定
- replay 导出、分类训练、offline replay 评测链路都可以复现
- Qwen 3B LoRA 已经能完整复现当前 weak policy 在 validation 上的行为模式

但这并不等价于：

- 已经证明 action policy 具备强泛化能力
- 已经可以直接进入 rollout / RL 主循环
- 已经可以把当前高分当成最终 agent 能力指标

当前能证明的是：

- weakly-coupled action policy offline 主线已经跑通
- 可以在 frozen verifier 条件下继续做更正式的 action policy 实验

## 10. 建议的下一步

1. 继续保持 frozen verifier 不动
2. 扩展 action replay 数据与 episode-level 诊断
3. 增加更严格的 offline replay 对照实验
4. 等 replay 主线足够稳后，再考虑更正式的 action-policy mainline 和后续 RL
