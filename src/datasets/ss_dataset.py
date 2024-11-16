from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH
import os
from tqdm import tqdm


class SSAudioOnlyDataset(BaseDataset):
    def __init__(self, part, *args, **kwargs):
        self._data_dir = ROOT_PATH / "data" / "dla_dataset" 

        mix_audio_dir = self._data_dir / "audio" / part / "mix"
        s1_audio_dir = self._data_dir / "audio" / part / "s1"
        s2_audio_dir = self._data_dir / "audio"/ part / "s2" 
        # video_dir = self._data_dir / "video"
        data = []
        dir_len = len(os.listdir(mix_audio_dir))
        i = 0
        for path in tqdm(Path(mix_audio_dir).iterdir(), total=dir_len):
            entry = {}
            if path.suffix in [".wav", ".mp3", ".flac"]:
                entry["mix_path"] = str(path)
                audio_len = self._calc_audio_len(entry["mix_path"])
                entry["mix_len"] = audio_len

                if s1_audio_dir and Path(s1_audio_dir).exists():
                    s1_path = Path(s1_audio_dir) / (path.stem + path.suffix)
                    if s1_path.exists():
                        entry["s1_path"] = str(s1_path)
                        entry["s1_len"] = audio_len

                if s2_audio_dir and Path(s2_audio_dir).exists():
                    s2_path = Path(s2_audio_dir) / (path.stem + path.suffix)
                    if s2_path.exists():
                        entry["s2_path"] = str(s2_path)
                        entry["s2_len"] = audio_len

            if len(entry) == 6:
                data.append(entry)
            i += 1
            if i == 10:
                break

        super().__init__(data, *args, **kwargs)


    def _calc_audio_len(self, audio_path):
        t_info = torchaudio.info(audio_path)
        return t_info.num_frames / t_info.sample_rate
