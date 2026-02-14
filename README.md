# Explicit Estimation of Magnitude and Phase Spectra in Parallel for High-Quality Speech Enhancement
### Ye-Xin Lu, Yang Ai, Zhen-Hua Ling
In our [paper](https://arxiv.org/abs/2305.13686), we proposed MP-SENet: a TF-domain monaural SE model with parallel magnitude and phase spectra denoising.<br>
A [long-version](https://arxiv.org/abs/2308.08926) MP-SENet was extended to the speech denoising, dereverberation, and bandwidth extension tasks.<br>
Audio samples can be found at the [demo website](http://yxlu-0102.github.io/MP-SENet).<br>
We provide our implementation as open source in this repository.

## Pre-requisites
1. Python >= 3.13
2. Clone this repository
3. Install dependencies via [uv](https://github.com/astral-sh/uv):
```
uv sync
```
4. Download and extract the [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/handle/10283/1942). Resample all wav files to 16kHz, and move the clean and noisy wavs to `VoiceBank+DEMAND/wavs_clean` and `VoiceBank+DEMAND/wavs_noisy`, respectively. You can also directly download the downsampled 16kHz dataset [here](https://drive.google.com/drive/folders/19I_thf6F396y5gZxLTxYIojZXC0Ywm8l).

## Training

Single GPU:
```bash
uv run python train.py
```

Multi-GPU via `torchrun` (all available GPUs):
```bash
uv run torchrun --nproc-per-node=gpu train.py
```

Multi-GPU with explicit GPU count:
```bash
uv run torchrun --nproc-per-node=2 train.py
```

### Training arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config.yaml` | Path to config file |
| `--checkpoint_path` | `cp_model` | Directory for checkpoints and logs |
| `--input_clean_wavs_dir` | `/work/VoiceBank+DEMAND/wav_clean` | Clean waveforms directory |
| `--input_noisy_wavs_dir` | `/work/VoiceBank+DEMAND/wav_noisy` | Noisy waveforms directory |
| `--input_training_file` | `/work/VoiceBank+DEMAND/training.txt` | Training file list |
| `--input_validation_file` | `/work/VoiceBank+DEMAND/test.txt` | Validation file list |
| `--training_epochs` | `400` | Total epochs |
| `--checkpoint_interval` | `5000` | Save checkpoint every N steps |
| `--validation_interval` | `5000` | Run validation every N steps |
| `--best_checkpoint_start_epoch` | `40` | Start saving best checkpoint after this epoch |

Checkpoints and a copy of `config.yaml` are saved to `--checkpoint_path`.
TensorBoard logs are in `<checkpoint_path>/logs/`.

### Resuming training

Training resumes automatically if checkpoints (`g_*` and `do_*`) exist in `--checkpoint_path`:
```bash
uv run torchrun --nproc-per-node=gpu train.py --checkpoint_path cp_model
```

### Monitoring

```bash
uv run tensorboard --logdir cp_model/logs
```

## Inference

```bash
uv run python inference.py --checkpoint_file cp_model/g_best
```

With PCS (Perceptual Contrast Stretching) post-processing:
```bash
uv run python inference.py --checkpoint_file cp_model/g_best --use_pcs
```

### Inference arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_file` | *(required)* | Path to generator checkpoint |
| `--input_noisy_wavs_dir` | `/work/VoiceBank+DEMAND/testset_noisy` | Input noisy waveforms |
| `--output_dir` | `../generated_files` | Output directory for enhanced waveforms |
| `--use_pcs` | `false` | Enable PCS post-processing |

The config is loaded automatically from the same directory as the checkpoint file.

## Evaluation

Compute metrics for a single run:
```bash
cd cal_metrics
uv run python cal_metrics_vb.py \
  --clean_wav_dir /work/VoiceBank+DEMAND/testset_clean \
  --noisy_wav_dir ../generated_files
```

Compare two runs (e.g. baseline vs PCS):
```bash
cd cal_metrics
uv run python cal_metrics_vb.py \
  --clean_wav_dir /work/VoiceBank+DEMAND/testset_clean \
  --noisy_wav_dir ../generated_baseline \
  --compare_dir ../generated_pcs
```

Output:
```
-------------------------------------------------
Metric     |   Baseline |        PCS |      Delta
-------------------------------------------------
PESQ       |     3.5905 |     3.6281 | +   0.0376
CSIG       |     4.7213 |     4.7191 | -   0.0022
CBAK       |     3.9147 |     3.6081 | -   0.3066
COVL       |     4.2346 |     4.2499 | +   0.0153
SSNR       |    10.7145 |     5.5693 | -   5.1452
STOI       |     0.9615 |     0.9614 | -   0.0001
-------------------------------------------------
```

## Model Structure
![model](Figures/model.png)

## Comparison with other SE models
![comparison](Figures/table.png)

## Acknowledgements
We referred to [HiFiGAN](https://github.com/jik876/hifi-gan), [NSPP](https://github.com/YangAi520/NSPP) 
and [CMGAN](https://github.com/ruizhecao96/CMGAN) to implement this.

## Citation
```
@inproceedings{lu2023mp,
  title={{MP-SENet}: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra},
  author={Lu, Ye-Xin and Ai, Yang and Ling, Zhen-Hua},
  booktitle={Proc. Interspeech},
  pages={3834--3838},
  year={2023}
}

@article{lu2023explicit,
  title={Explicit estimation of magnitude and phase spectra in parallel for high-quality speech enhancement},
  author={Lu, Ye-Xin and Ai, Yang and Ling, Zhen-Hua},
  journal={Neural Networks},
  volume = {189},
  pages = {107562},
  year={2025}
}
```
