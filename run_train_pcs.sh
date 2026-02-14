#!/bin/bash
uv run torchrun --standalone --nproc_per_node=gpu train.py \
  --config config_pcs.yaml \
  --checkpoint_path cp_model_pcs \
  --training_epochs 400
