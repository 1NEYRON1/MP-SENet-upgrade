import os
import argparse
import numpy as np
from torchcodec.decoders import AudioDecoder
from compute_metrics import compute_metrics
from rich.progress import track

METRIC_NAMES = ['PESQ', 'CSIG', 'CBAK', 'COVL', 'SSNR', 'STOI']


def calc_avg_metrics(clean_dir, enhanced_dir, sampling_rate):
    indexes = sorted(os.listdir(clean_dir))
    num = len(indexes)
    metrics_total = np.zeros(6)
    for index in track(indexes, description=f"  {os.path.basename(enhanced_dir)}"):
        clean_wav = os.path.join(clean_dir, index)
        enhanced_wav = os.path.join(enhanced_dir, index)
        clean = AudioDecoder(clean_wav, sample_rate=sampling_rate, num_channels=1).get_all_samples().data.squeeze(0).numpy()
        enhanced = AudioDecoder(enhanced_wav, sample_rate=sampling_rate, num_channels=1).get_all_samples().data.squeeze(0).numpy()

        metrics = compute_metrics(clean, enhanced, sampling_rate, 0)
        metrics_total += np.array(metrics)

    return metrics_total / num


def print_single(metrics_avg):
    for name, val in zip(METRIC_NAMES, metrics_avg):
        print(f'{name:10s} {val:.4f}')


def print_comparison(metrics_a, metrics_b, label_a='Baseline', label_b='PCS'):
    w = max(len(label_a), len(label_b), 10)
    header = f"{'Metric':10s} | {label_a:>{w}s} | {label_b:>{w}s} | {'Delta':>10s}"
    sep = '-' * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, va, vb in zip(METRIC_NAMES, metrics_a, metrics_b):
        delta = vb - va
        sign = '+' if delta >= 0 else '-'
        print(f'{name:10s} | {va:{w}.4f} | {vb:{w}.4f} | {sign} {abs(delta):8.4f}')
    print(sep)


def main(h):
    metrics_baseline = calc_avg_metrics(h.clean_wav_dir, h.noisy_wav_dir, h.sampling_rate)

    if h.compare_dir:
        metrics_compare = calc_avg_metrics(h.clean_wav_dir, h.compare_dir, h.sampling_rate)
        label_a = os.path.basename(h.noisy_wav_dir.rstrip('/')) or 'Baseline'
        label_b = os.path.basename(h.compare_dir.rstrip('/')) or 'Compare'
        print_comparison(metrics_baseline, metrics_compare, label_a, label_b)
    else:
        print_single(metrics_baseline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_rate', default=16000, type=int)
    parser.add_argument('--clean_wav_dir', required=True)
    parser.add_argument('--noisy_wav_dir', required=True)
    parser.add_argument('--compare_dir', default=None, help='Second enhanced dir for comparison table')

    h = parser.parse_args()

    main(h)
