#!/bin/bash
uv run python inference_pcs.py \
  --checkpoint_file /work/MP-SENet-upgrade/cp_model/g_best \
  --input_clean_wavs_dir /work/VoiceBank+DEMAND/testset_clean \
  --gamma 0.5 1.0 1.5
