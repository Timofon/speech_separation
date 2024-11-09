from pathlib import Path

import torchaudio

from datasets import load_dataset
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class SSDataset(BaseDataset):
    def __init__(self, mix_audio_dir, s1_audio_dir=None, s2_audio_dir=None, *args, **kwargs):
        data = []
        for path in Path(mix_audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["mix_path"] = str(path)
                entry["audio_len"] = self._calc_audio_len(entry["mix_path"])

                if s1_audio_dir and Path(s1_audio_dir).exists():
                    s1_path = Path(s1_audio_dir) / (path.stem + path.suffix)
                    if s1_path.exists():
                        entry["s1_path"] = str(s1_path)
                        entry["s1_len"] = self._calc_audio_len(entry["s1_path"])

                if s2_audio_dir and Path(s2_audio_dir).exists():
                    s2_path = Path(s2_audio_dir) / (path.stem + path.suffix)
                    if s2_path.exists():
                        entry["s2_path"] = str(s2_path)
                        entry["s2_len"] = self._calc_audio_len(entry["s2_path"])

            if len(entry) == 6:
                data.append(entry)

        super().__init__(data, *args, **kwargs)  


    def _calc_audio_len(self, audio_path):
        t_info = torchaudio.info(audio_path)
        return t_info.num_frames / t_info.sample_rate