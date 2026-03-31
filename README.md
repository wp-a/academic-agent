# Academic Agent

面向 `deep_research_review_v2` 的可训练 evidence-seeking agent 原型仓库。

当前仓库已经不再停留在最早的 demo scaffold，而是围绕“受限证据环境中的模块化 agent”推进。当前重点不是直接训练一个完整 reviewer agent，而是先把下面几条可单独验证的链路做稳：

- `verifier`：在受限文档集合内对证据相关性和立场进行判别与排序
- `action policy`：在 frozen verifier 提供的状态下，学习 `search / quote_evidence / stop`
- `stop policy`：在 frozen verifier 提供的状态下，学习 `should_stop = yes / no`
- `restricted env`：用于离线 replay、ranking 评测和后续更正式主循环接入

当前已经完成：

- SciFact `decomposed verifier` 路线
- frozen verifier 下的 restricted ranking 评测
- weakly-coupled replay 的 action / stop 数据导出
- hard replay 的 action / stop 数据导出
- action policy 小分类 baseline
- `Qwen2.5-3B-Instruct` LoRA action policy 主线
- `Qwen2.5-3B-Instruct` LoRA stop policy 主线
- easy / hard 两套 action-level 与 episode-level offline replay 评测

当前明确不做：

- 不直接训练完整 reviewer head
- 不进入 online rollout / RL 主循环
- 不在当前阶段混 FEVER/HoVer 联训到主线上

## 0. 导航

- 公开仓库首页说明：本文档
- 复现实验说明：`docs/reproduce.md`
- 训练设计文档：`docs/plans/2026-03-27-agent-training-design.md`
- 训练实施文档：`docs/plans/2026-03-27-agent-training.md`

## 1. 当前项目目标

目标是构建一个在受限证据环境中真正有用的 evidence-seeking agent，而不是只做一个教学版的最小分类器。

当前的工程策略是：

1. 先冻结一个足够稳定的 verifier
2. 基于 frozen verifier 导出 action policy / stop policy replay 数据
3. 先做 offline imitation / replay 验证
4. 等 action policy 与 stop policy 和 restricted env 的接口跑稳，再讨论更正式的 agent loop

## 2. 当前主线状态

### 2.1 Verifier 主线

当前冻结的 relevance baseline 是：

- `v7_pairwise_margin + full_document`

它是当前 frozen verifier 的主线基线，用于：

- restricted env 文档揭示顺序
- SciFact weak replay / hard replay 数据导出
- offline replay 环境中的 verifier side signal

当前 stance 头保持为 decomposed verifier 结构的一部分，但本阶段不继续扩展它。

### 2.2 Action Policy 主线

当前 action policy 是一个三分类任务：

- `search`
- `quote_evidence`
- `stop`

输入是 verifier 驱动环境状态的 `state_text`，输出是离散 `action_type`。

当前真实 `state_text` 由 `train_agent/rl/restricted_retrieval.py` 里的 `RestrictedRetrievalState.to_text()` 生成，包含：

- `claim`
- `current observation`
- `history`
- `verifier summary`
- `evidence`
- `action space`

当前已经跑通三条线：

- 小分类 baseline：`prajjwal1/bert-tiny`
- easy replay 主力 baseline：`Qwen2.5-3B-Instruct` + LoRA
- hard replay 主力 baseline：`Qwen2.5-3B-Instruct` + LoRA

### 2.3 Stop Policy 主线

当前 stop policy 是一个二分类任务：

- `no`
- `yes`

输入也是 verifier 驱动环境状态的 `state_text`，输出是离散 `should_stop`。

当前已经跑通：

- SciFact weak replay stop 数据导出
- SciFact hard replay stop 数据导出
- `Qwen2.5-3B-Instruct` + LoRA stop policy 四卡训练

当前最新验证结果：

- weak replay validation `accuracy = 1.0`
- weak replay validation `macro_f1 = 1.0`
- hard replay validation `accuracy = 0.998563`
- hard replay validation `macro_f1 = 0.997962`

### 2.4 评测方式

当前 action policy 不是看生成文本，而是看两层指标：

1. step-level classification
   - `accuracy`
   - `macro_f1`
   - per-action `precision / recall / f1`
   - confusion matrix
   - gold / prediction distribution
   - per-example prediction / error export
2. episode-level offline replay
   - `action_agreement`
   - `average_steps`
   - `stop_precision / stop_recall`
   - `quote_evidence_hit_rate`
   - `success_rate`
   - `early_stop_rate`

当前除了 weak replay 以外，还支持带 lexical distractor 的 hard replay：

- 通过 `train_agent/data/adapters/scifact_hard.py` 为每个 episode 追加词汇重叠但非 gold 的干扰文档
- 通过 `train_agent/scripts/export_scifact_hard_replay_data.py` 导出更保守 teacher 的 action / stop 数据
- 通过 `train_agent/scripts/eval_action_policy_offline_replay.py --num_distractor_docs N --reference_policy_type conservative --post_quote_search_budget 1` 在 offline replay 中打开与 hard 导出 teacher 对齐的评测

## 3. 目录结构

### 3.1 训练与数据

- `train_agent/data/`
  - 数据 schema 与数据适配逻辑
- `train_agent/trajectories/`
  - trajectory schema 与历史导出逻辑
- `train_agent/models/`
  - frozen verifier / frozen action policy / frozen stop policy 推理封装
- `train_agent/trainers/`
  - `train_verifier.py`
  - `train_action_policy.py`
- `train_agent/eval/`
  - verifier metrics
  - restricted ranking
  - action policy metrics
- `train_agent/scripts/`
  - SciFact verifier 数据导出
  - easy replay action / stop 数据导出
  - hard replay action / stop 数据导出
  - step-level classification 评测与错例导出
  - offline replay 评测
  - inference / utility 脚本
- `train_agent/rl/`
  - restricted retrieval environment

### 3.2 文档

- `docs/plans/2026-03-27-agent-training-design.md`
- `docs/plans/2026-03-27-agent-training.md`
- `docs/plans/2026-03-27-restricted-retrieval-rl-environment.md`
- `SESSION_STATE.md`

## 4. 环境约束

当前默认环境：

- 宿主机路径：`/public/localUsers/wangpengv2/AcademicSubmission`
- 容器路径：`/mnt/AcademicSubmission`
- 容器：`2e230dfba7c5`
- 训练方式：单机 `4 x V100-SXM2-32GB`
- 精度：`fp16`
- 分布式：`torchrun` / DDP

当前明确约束：

- 不使用 `bf16`
- 不使用 `flash-attn2`
- 不使用 `deepspeed`

所有 Python / 数据处理 / 训练 / 评测命令都应通过容器执行：

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && ..."
```

## 5. 代理与 Hugging Face 访问

容器内访问外网时，需要显式透传代理：

```sh
export HTTP_PROXY=http://172.17.0.1:17898
export HTTPS_PROXY=http://172.17.0.1:17898
export NO_PROXY=localhost,127.0.0.1
```

对于大模型下载，建议先单进程预拉取 snapshot，再启动 `torchrun`，避免多卡 rank 重复下载导致超时。

## 6. 当前已验证的数据与产物

### 6.1 SciFact easy replay 数据

当前 replay 数据目录：

- `data/processed/scifact_action_policy_v1/`
- `data/processed/scifact_stop_policy_v1/`

当前导出设置：

- frozen verifier：`outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin`
- doc aggregation：`full_document`
- max steps：`4`

当前数据规模：

- train：`957` episodes，`2974` action examples
- validation：`338` episodes，`1046` action examples

### 6.2 SciFact hard replay 数据

当前 hard replay 数据目录：

- `data/processed/scifact_hard_replay_v1/`

当前导出设置：

- frozen verifier：`outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin`
- doc aggregation：`full_document`
- max steps：`5`
- lexical distractors：`3`
- post quote search budget：`1`

当前数据规模：

- train action：`957` episodes，`3940` examples，平均 `4.117032` steps
- validation action：`338` episodes，`1392` examples，平均 `4.118343` steps
- validation action 分布：`quote_evidence=0.238506`，`search=0.533046`，`stop=0.228448`
- validation stop 分布：`no=0.771552`，`yes=0.228448`

### 6.3 Verifier 产物

当前 frozen verifier 主线：

- `outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin`

它是 easy / hard replay 导出与 offline replay 的共同依赖。

### 6.4 Action policy 模型输出

小分类 baseline：

- `outputs/action_policy_scifact_bert_tiny_v1`

Qwen LoRA easy replay 主线：

- `outputs/action_policy_scifact_qwen25_3b_lora_v1`

Qwen LoRA hard replay 主线：

- `outputs/action_policy_scifact_hard_qwen25_3b_lora_v1`
- validation `accuracy = 0.997845`
- validation `macro_f1 = 0.997776`

### 6.5 Stop policy 模型输出

Qwen LoRA easy replay 主线：

- `outputs/stop_policy_scifact_qwen25_3b_lora_v1`

Qwen LoRA hard replay 主线：

- `outputs/stop_policy_scifact_hard_qwen25_3b_lora_v1`
- validation `accuracy = 0.998563`
- validation `macro_f1 = 0.997962`

### 6.6 Joint offline replay 输出

weak replay joint 输出：

- `outputs/joint_policy_scifact_qwen25_3b_lora_v1`

hard replay joint 输出：

- `outputs/joint_policy_scifact_hard_qwen25_3b_lora_v1`
- validation `success_rate = 0.97929`
- validation `action_agreement = 0.998564`
- validation `stop_precision = 0.996865`
- validation `stop_recall = 0.996865`
- validation `quote_evidence_hit_rate = 1.0`
- validation `reference_policy_type = conservative`
- validation `post_quote_search_budget = 1`

## 7. 关键命令

### 7.1 导出 SciFact easy replay action 数据

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export HTTP_PROXY=http://172.17.0.1:17898 && export HTTPS_PROXY=http://172.17.0.1:17898 && export NO_PROXY=localhost,127.0.0.1 && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.scripts.export_scifact_action_policy_data --verifier_model_name_or_path outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin --output_dir data/processed/scifact_action_policy_v1 --max_steps 4 --doc_aggregation full_document --aggregation_top_k 3 --max_length 384 --batch_size 8"
```

### 7.2 训练小分类 baseline

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export HTTP_PROXY=http://172.17.0.1:17898 && export HTTPS_PROXY=http://172.17.0.1:17898 && export NO_PROXY=localhost,127.0.0.1 && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.trainers.train_action_policy --train_file data/processed/scifact_action_policy_v1/scifact_action_policy_train.jsonl --eval_file data/processed/scifact_action_policy_v1/scifact_action_policy_validation.jsonl --model_name_or_path prajjwal1/bert-tiny --output_dir outputs/action_policy_scifact_bert_tiny_v1 --max_length 256 --num_train_epochs 5 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --learning_rate 2e-4 --logging_steps 20 --eval_steps 40 --save_steps 40 --gradient_accumulation_steps 1 --attn_implementation sdpa"
```

### 7.3 预下载 Qwen2.5-3B-Instruct

```sh
docker exec -i 2e230dfba7c5 bash -lc "export HTTP_PROXY=http://172.17.0.1:17898 && export HTTPS_PROXY=http://172.17.0.1:17898 && export NO_PROXY=localhost,127.0.0.1 && export HF_HUB_DOWNLOAD_TIMEOUT=120 && export HF_HUB_ETAG_TIMEOUT=120 && /root/miniconda3/bin/python - <<\"PY\"
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id=\"Qwen/Qwen2.5-3B-Instruct\", max_workers=1, allow_patterns=[\"*.json\", \"*.safetensors\", \"tokenizer*\", \"merges.txt\", \"vocab.json\", \"*.model\", \"*.py\"]))
PY"
```

### 7.4 训练 Qwen2.5-3B-Instruct LoRA action policy

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export CUDA_VISIBLE_DEVICES=0,1,2,3 && /root/miniconda3/bin/torchrun --nproc_per_node=4 -m train_agent.trainers.train_action_policy --train_file data/processed/scifact_action_policy_v1/scifact_action_policy_train.jsonl --eval_file data/processed/scifact_action_policy_v1/scifact_action_policy_validation.jsonl --model_name_or_path /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1 --output_dir outputs/action_policy_scifact_qwen25_3b_lora_v1 --max_length 512 --num_train_epochs 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --learning_rate 1e-4 --logging_steps 10 --eval_steps 50 --save_steps 50 --attn_implementation sdpa --use_lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --lora_target_modules q_proj,k_proj,v_proj,o_proj --lora_modules_to_save score --gradient_checkpointing"
```

### 7.5 运行 action policy step-level 评测

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.scripts.eval_action_policy_predictions --model_dir outputs/action_policy_scifact_bert_tiny_v1 --eval_file data/processed/scifact_action_policy_v1/scifact_action_policy_validation.jsonl --output_path outputs/action_policy_scifact_bert_tiny_v1/step_eval_validation.json --errors_output_path outputs/action_policy_scifact_bert_tiny_v1/step_eval_validation_errors.json --max_length 256 --batch_size 32"
```

### 7.6 运行 action policy offline replay 评测

小模型：

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.scripts.eval_action_policy_offline_replay --policy_model_dir outputs/action_policy_scifact_bert_tiny_v1 --verifier_model_name_or_path outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin --output_path outputs/action_policy_scifact_bert_tiny_v1/offline_replay_validation.json --split validation --max_steps 4 --policy_max_length 256 --policy_batch_size 32 --verifier_max_length 384 --verifier_batch_size 8 --doc_aggregation full_document --aggregation_top_k 3"
```

Qwen LoRA：

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.scripts.eval_action_policy_offline_replay --policy_model_dir outputs/action_policy_scifact_qwen25_3b_lora_v1 --verifier_model_name_or_path outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin --output_path outputs/action_policy_scifact_qwen25_3b_lora_v1/offline_replay_validation.json --split validation --max_steps 4 --policy_max_length 512 --policy_batch_size 8 --verifier_max_length 384 --verifier_batch_size 8 --doc_aggregation full_document --aggregation_top_k 3"
```

联合 `action policy + stop policy`：

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.scripts.eval_action_policy_offline_replay --policy_model_dir outputs/action_policy_scifact_qwen25_3b_lora_v1 --stop_model_dir outputs/stop_policy_scifact_qwen25_3b_lora_v1 --verifier_model_name_or_path outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin --output_path outputs/joint_policy_scifact_qwen25_3b_lora_v1/offline_replay_validation.json --split validation --max_steps 4 --policy_max_length 512 --policy_batch_size 8 --stop_max_length 512 --stop_batch_size 8 --verifier_max_length 384 --verifier_batch_size 8 --doc_aggregation full_document --aggregation_top_k 3"
```

### 7.7 导出 SciFact hard replay 数据

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export HTTP_PROXY=http://172.17.0.1:17898 && export HTTPS_PROXY=http://172.17.0.1:17898 && export NO_PROXY=localhost,127.0.0.1 && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.scripts.export_scifact_hard_replay_data --verifier_model_name_or_path outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin --output_dir data/processed/scifact_hard_replay_v1 --max_steps 5 --num_distractor_docs 3 --post_quote_search_budget 1 --doc_aggregation full_document --aggregation_top_k 3 --max_length 384 --batch_size 8"
```

### 7.8 运行 hard joint offline replay 评测

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && export CUDA_VISIBLE_DEVICES=0 && /root/miniconda3/bin/python -m train_agent.scripts.eval_action_policy_offline_replay --policy_model_dir outputs/action_policy_scifact_hard_qwen25_3b_lora_v1 --stop_model_dir outputs/stop_policy_scifact_hard_qwen25_3b_lora_v1 --verifier_model_name_or_path outputs/verifier_scifact_deberta_v3_large_relevance_v7_pairwise_margin --output_path outputs/joint_policy_scifact_hard_qwen25_3b_lora_v1/offline_replay_validation_hard.json --split validation --max_steps 5 --policy_max_length 512 --policy_batch_size 8 --stop_max_length 512 --stop_batch_size 8 --verifier_max_length 384 --verifier_batch_size 8 --doc_aggregation full_document --aggregation_top_k 3 --num_distractor_docs 3 --reference_policy_type conservative --post_quote_search_budget 1"
```

## 8. 当前结论

### 8.1 关于 verifier

- relevance 主线冻结在 `v7_pairwise_margin + full_document`
- 当前不继续做 verifier loss 试验

### 8.2 关于 action policy

当前已经具备开始更正式 action policy 主线实验的工程条件：

- frozen verifier 接口稳定
- replay 数据可稳定导出
- 小模型与 3B LoRA 模型都能完成 step-level 训练
- offline replay 评测链路已打通

但需要明确：

- 当前 easy replay supervision 仍来自 weak policy
- 当前高分首先证明的是接口和 imitation pipeline 稳定
- 还不能把当前高分直接等价为最终 agent 能力
- 当前 `Qwen action + Qwen stop` 在 easy replay 上已经非常接近饱和
- hard replay 已经把 validation 平均步数拉到 `4.12`，`search` 比例拉到 `0.533`
- hard replay 上 step-level 分类仍接近饱和，但对齐 `ConservativeReplayPolicy` 后 joint offline replay 也基本复现了 hard teacher
- 这说明前一版 `0.814788 / 0.552265` 的主要来源是 eval reference teacher 与导出 teacher 不一致，而不是 hard student 本身显著退化

### 8.3 关于 hard replay 结果的当前解释

- hard replay 数据导出使用的是 `ConservativeReplayPolicy`
- `eval_action_policy_offline_replay.py` 现已支持 `--reference_policy_type {weak,conservative}` 与 `--post_quote_search_budget`
- 当前 hard joint 指标应使用 teacher-aligned 的 conservative reference 结果来解读
- 对齐后 `action_agreement = 0.998564`、`stop_recall = 0.996865`
- 旧的 `0.814788 / 0.552265` 是 reference mismatch 造成的历史结果，不应继续作为当前主结论

## 9. 下一步建议

建议按下面顺序继续推进：

1. 保持 frozen verifier 不动
2. 直接对 hard replay 中剩余的少量 mismatch episode 做更严格的 disagreement / stop 失败诊断
3. 然后继续扩展 action policy / stop policy replay 数据和评测切面
4. 等 joint replay 在更强数据上仍稳定后，再考虑更正式的 agent loop

当前不建议直接做：

- online RL
- rollout 主循环
- verifier / action policy 同时联动重训

## 10. 备注

仓库里已有一些历史脚本和早期最小骨架；当前更可信的运行入口以 `train_agent/` 目录、`docs/plans/` 文档和本 README 为准。
