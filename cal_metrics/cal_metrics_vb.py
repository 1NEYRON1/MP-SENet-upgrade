import os
import argparse
import numpy as np
from torchcodec.decoders import AudioDecoder
from compute_metrics import compute_metrics
from rich.progress import track


def main(h):
    indexes = sorted(os.listdir(h.clean_wav_dir))
    num = len(indexes)
    metrics_total = np.zeros(6)
    for index in track(indexes):
        clean_wav = os.path.join(h.clean_wav_dir, index)
        noisy_wav = os.path.join(h.noisy_wav_dir, index)
        clean = AudioDecoder(clean_wav, sample_rate=h.sampling_rate, num_channels=1).get_all_samples().data.squeeze(0).numpy()
        noisy = AudioDecoder(noisy_wav, sample_rate=h.sampling_rate, num_channels=1).get_all_samples().data.squeeze(0).numpy()

        metrics = compute_metrics(clean, noisy, h.sampling_rate, 0)
        metrics = np.array(metrics)
        metrics_total += metrics

    metrics_avg = metrics_total / num
    print('pesq: ', metrics_avg[0], 'csig: ', metrics_avg[1], 'cbak: ', metrics_avg[2],
          'covl: ', metrics_avg[3], 'ssnr: ', metrics_avg[4], 'stoi: ', metrics_avg[5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_rate', default=16000, type=int)
    parser.add_argument('--clean_wav_dir', required=True)
    parser.add_argument('--noisy_wav_dir', required=True)

    h = parser.parse_args()

    main(h)
