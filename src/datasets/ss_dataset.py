from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class SSAudioOnlyDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self._data_dir = ROOT_PATH / "data" / "dataset_ss"

        mix_audio_dir = self._data_dir / "audio" / "mix"
        s1_audio_dir = self._data_dir / "audio" / "s1"
        s2_audio_dir = self._data_dir / "audio" / "s2" 
        # video_dir = self._data_dir / "video"
        data = []
        for path in Path(mix_audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".flac"]:
                entry["mix_path"] = str(path)
                entry["mix_len"] = self._calc_audio_len(entry["mix_path"])

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