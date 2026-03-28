# AcademicSubmission

Minimal evidence-seeking agent scaffold for `deep_research_review_v2`.

Current scope:

- trajectory recording in JSONL
- supervised data export for `next_action` and `stopping`
- LoRA SFT training script compatible with single-node 4x V100, fp16, DDP via `torchrun`

The project intentionally starts small so the agent loop can be trained before adding online RL or heavier verifier stacks.

## Layout

- `deep_research_review_v2/trajectory.py`: trajectory schema and recorder
- `deep_research_review_v2/export_sft_data.py`: export trajectories to SFT JSONL
- `deep_research_review_v2/train_sft.py`: LoRA SFT entrypoint
- `scripts/create_demo_trajectories.py`: generate demo data

## Container commands

Generate demo trajectories:

```sh
docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && /root/miniconda3/bin/conda run -n base python -m scripts.create_demo_trajectories --output data/demo_trajectories.jsonl'
```

Export SFT data:

```sh
docker exec -i 2e230dfba7c5 bash -lc 'cd /mnt/AcademicSubmission && /root/miniconda3/bin/conda run -n base python -m deep_research_review_v2.export_sft_data --input data/demo_trajectories.jsonl --output_dir data/exports'
```

Train next-action policy:

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/bin/conda run -n base torchrun --nproc_per_node=4 -m deep_research_review_v2.train_sft --task next_action --train_file data/exports/next_action_train.jsonl --eval_file data/exports/next_action_eval.jsonl --model_name_or_path meta-llama/Llama-3.2-1B --output_dir outputs/next_action_lora --fp16"
```

Train stopping policy:

```sh
docker exec -i 2e230dfba7c5 bash -lc "cd /mnt/AcademicSubmission && CUDA_VISIBLE_DEVICES=0,1,2,3 /root/miniconda3/bin/conda run -n base torchrun --nproc_per_node=4 -m deep_research_review_v2.train_sft --task stopping --train_file data/exports/stopping_train.jsonl --eval_file data/exports/stopping_eval.jsonl --model_name_or_path meta-llama/Llama-3.2-1B --output_dir outputs/stopping_lora --fp16"
```
