import argparse
import pathlib
from pathlib import Path
from tqdm import tqdm
import os
import torchaudio

from src.metrics.utils import calc_si_sdr, calc_si_sdri, calc_si_snri

def load_audio(path, target_sr=16000):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

def calc_metrics(mix_dir, s1_pred_dir, s2_pred_dir, s1_gt_dir=None, s2_gt_dir=None):
    if s1_gt_dir is None and s2_gt_dir is None:
        print("Need at least one groundtruth directory")
    si_sdri = 0
    si_snri = 0
    file_counter = 0
    dir_len = len(os.listdir(mix_dir))
    for path in tqdm(Path(mix_dir).iterdir(), total=dir_len):
        if path.suffix in [".wav", ".mp3", ".flac"]:
            s1_pred_path = Path(s1_pred_dir) / (path.stem + path.suffix)
            s2_pred_path = Path(s2_pred_dir) / (path.stem + path.suffix)
            s1_gt_path = Path(s1_gt_dir) / (path.stem + path.suffix)
            s2_gt_path = Path(s2_gt_dir) / (path.stem + path.suffix)

            mix_audio = load_audio(path)
            s1_pred_audio = load_audio(s1_pred_path)
            s2_pred_audio = load_audio(s2_pred_path)
            s1_gt_audio = load_audio(s1_gt_path)
            s2_gt_audio = load_audio(s2_gt_path)

            # SI-SDRI
            s1_true = calc_si_sdri(s1_pred_audio, s1_gt_audio, mix_audio)
            s2_true = calc_si_sdri(s2_pred_audio, s2_gt_audio, mix_audio)

            s1_permuted = calc_si_sdri(s1_pred_audio, s2_gt_audio, mix_audio)
            s2_permuted = calc_si_sdri(s2_pred_audio, s1_gt_audio, mix_audio)

            si_sdri += max((s1_true + s2_true) / 2, (s1_permuted + s2_permuted) / 2)

            # SI-SNRI
            s1_true = calc_si_snri(s1_pred_audio, s1_gt_audio, mix_audio)
            s2_true = calc_si_snri(s2_pred_audio, s2_gt_audio, mix_audio)

            s1_permuted = calc_si_snri(s1_pred_audio, s2_gt_audio, mix_audio)
            s2_permuted = calc_si_snri(s2_pred_audio, s1_gt_audio, mix_audio)

            si_snri += max((s1_true + s2_true) / 2, (s1_permuted + s2_permuted) / 2)

            file_counter += 1

    print(f"SI-SDRI: {(si_sdri / file_counter)}")
    print(f"SI-SNRI: {(si_snri / file_counter)}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--mix_dir", required=True, type=str, help="Path to directory with mixed_audios")
    args.add_argument("--s1_pred_dir", required=True, type=str, help="Path to directory with s1 predictions")
    args.add_argument("--s2_pred_dir", required=True, type=str, help="Path to directory with s2 predictions")
    args.add_argument("--s1_gt_dir", required=False, type=str, help="Path to directory with s1 ground truths")
    args.add_argument("--s2_gt_dir", required=False, type=str, help="Path to directory with s2 ground truths")

    args = args.parse_args()
    calc_metrics(mix_dir=args.mix_dir, s1_pred_dir=args.s1_pred_dir, s2_pred_dir=args.s2_pred_dir, s1_gt_dir=args.s1_gt_dir, s2_gt_dir=args.s2_gt_dir)
