from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import audio_conv as ac
import dataset.data_augmentation as da


class Copyright40Dataset(Dataset):
    """_summary_: DataLoader for Pair Copyright40 Dataset
        copyright40.csv contains Infringed work(source song) and Defending work(target song)
        [file structure]
        data
        ㄴCopyright40
            ㄴCopyrightCases
                ㄴFull .mp3 (39 pairs | 78 songs)
                    ㄴ00a{side}_{artist}_{song_title}_F.mp3
                    ㄴ00b{side}_{artist}_{song_title}_F.mp3
                ㄴMelody .mp3 (39 pairs | 78 songs) - same case, only melodyline
        
    Attr:
        pair_csv_path: metadata path of csv file
        dpath: data path - path to data / "copyright40"
        inp_rep: input representation
        crop_len: length of input time frame
        split: train, validate, test division
        cache_rep: whether or not to cash/use cached input feature
        cache_dir: directory of cache
    """
    def __init__(self, pair_csv_path, dpath, inp_rep, crop_len, division="Full", cache_rep=True, cache_dir=None, crop=False):
        self.pair_csv_path = Path(pair_csv_path)
        self.dpath = Path(dpath) / division
        self.inp_rep = inp_rep
        self.crop_len = crop_len
        self.src_paths = [] # source (Infringed)
        self.tgt_paths = [] # target (Defending)
        self.work_ids = [] # work_id = pair label
        self.src_perf_ids = [] # perf_id = individual music label
        self.tgt_perf_ids = []
        self.decisions = [] # similar or not
        self.cache_rep = cache_rep
        self.crop = crop

        if cache_dir is None: # chromagram 미리 캐싱 해두는 directory
            self.cache_dir = Path("cache") / "{}_cache".format(self.inp_rep.lower()) / self.dpath.name # default cache_dir
        else:
            self.cache_dir = Path(cache_dir)

        if self.cache_rep:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # read csv
        copy_df = pd.read_csv(self.pair_csv_path) # header(column 이름) exists
        copy_df = copy_df.dropna(how="all").sort_values("stimuli ID").reset_index(drop=True) # remove empty rows

        for stimuli_ID in sorted(copy_df["stimuli ID"].unique()):
            stimuli_ID = int(stimuli_ID)
            self.work_ids.append(stimuli_ID)

            self.src_perf_ids.append(stimuli_ID * 2) # 00, 02, 04, ...
            self.tgt_perf_ids.append(stimuli_ID * 2 + 1)

            decision = copy_df.loc[copy_df["stimuli ID"] == stimuli_ID, "Court Decision (num)"].iloc[0]
            self.decisions.append(int(decision))

        self.src_paths = ['' for i in range(len(self.work_ids))]
        self.tgt_paths = ['' for i in range(len(self.work_ids))]
        for fpath in self.dpath.iterdir():
            if fpath.is_file() and fpath.suffix.lower() == ".mp3":
                fstimuli_id, side = self.get_pair_info(fpath)

                if side == 'a':
                    self.src_paths[fstimuli_id] = fpath
                elif side == 'b':
                    self.tgt_paths[fstimuli_id] = fpath

        # sanity check
        assert len(self.src_paths) == len(self.tgt_paths)
        assert len(self.src_paths) == len(self.work_ids)

        for src_path, tgt_path, stimuli_id in zip(self.src_paths, self.tgt_paths, self.work_ids):
            src_id, src_side = self.get_pair_info(src_path)
            tgt_id, tgt_side = self.get_pair_info(tgt_path)

            assert src_id == stimuli_id
            assert tgt_id == stimuli_id
            assert src_side == "a"
            assert tgt_side == "b"

    def __len__(self):
        return len(self.src_paths)
    
    def get_pair_info(self, file_path):
        fname = Path(file_path).stem # 파일 이름

        match = re.match(r"(\d+)([ab])_", fname)
        if match is None:
            raise ValueError("Different filename format: {}".format(fname))

        stimuli_id = int(match.group(1)) # a, b 앞의 숫자가 work_id와 동일합니다.
        side = match.group(2) # a = source, b = target

        return stimuli_id, side

    def test_crop_1(self, raw_inp): # pad at back / cut middle
        bins, cur_len = raw_inp.shape

        if self.crop_len > cur_len: # zero-pad
            _pad = self.crop_len - cur_len

            return F.pad(raw_inp, (0, _pad), mode="constant", value=raw_inp.min().item())
        else:
            start = (cur_len - self.crop_len) // 2
            return raw_inp[:, start:self.crop_len + start]


    def test_crop_nseg(self, raw_inp, nseg=3):
        bins, cur_len = raw_inp.shape

        if self.crop_len > cur_len: # pad 
            _pad = self.crop_len - cur_len

            return F.pad(raw_inp, (0, _pad), mode="constant", value=raw_inp.min().item())
        else:
            starts = torch.linspace(0, cur_len - self.crop_len, steps=nseg).round().long() # long - 64 bit int
            crops = [raw_inp[:, start:start + self.crop_len] for start in starts]

            return torch.stack(crops, dim=0)

    
    def get_feature(self, feature_path, performance_id):
        if self.inp_rep == "CQT":
            get_feature = ac.get_cqt_spec
        elif self.inp_rep == "HPCP":
            get_feature = ac.get_hpcp
        else:
            raise ValueError("Unknown input representation: {}".format(self.inp_rep))

        if not self.cache_rep: # have to calculate
            feature = get_feature(feature_path)
            return torch.from_numpy(feature).float()

        cache_path = self.cache_dir / "{}.pt".format(performance_id)

        if cache_path.exists():
            return torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )
        else: # Not cached yet
            feature = get_feature(feature_path)
            feature = torch.from_numpy(np.asarray(feature, dtype=np.float32))
            torch.save(feature, cache_path)

            return feature
    
    def __getitem__(self, pidx):
        src_path, tgt_path, = self.src_paths[pidx], self.tgt_paths[pidx]
        work_id = self.work_ids[pidx]
        decision = self.decisions[pidx]
        src_pid, tgt_pid = self.src_perf_ids[pidx], self.tgt_perf_ids[pidx]
        src_cid, tgt_cid = "{:02d}a".format(work_id), "{:02d}b".format(work_id) # this becomes name for saving cache
        
        src_feature, tgt_feature = self.get_feature(src_path, src_cid), self.get_feature(tgt_path, tgt_cid)

        fs_input, ft_input = self.test_crop_1(src_feature), self.test_crop_1(tgt_feature) # numpy-fixed source / target
        fs_input = da.mirror_pitch_axis(fs_input)
        ft_input = da.mirror_pitch_axis(ft_input)
        tfs_input, tft_input = torch.as_tensor(fs_input, dtype=torch.float32), torch.as_tensor(ft_input, dtype=torch.float32)
        
        tfs_input = tfs_input.unsqueeze(0)
        if self.crop:
            tfs_input = da.cut_hpcp(fs_input)
        tft_input = tft_input.unsqueeze(0)

        tsrc_pid, ttgt_pid = torch.tensor(src_pid, dtype=torch.long), torch.tensor(tgt_pid, dtype=torch.long)
        twork_id = torch.tensor(work_id, dtype=torch.long)
        tdecision = torch.tensor(decision, dtype=torch.long)

        return {
            "work_id": twork_id, # used for MRR | MAP calculation
            "src_perf_id": tsrc_pid,
            "tgt_perf_id": ttgt_pid,
            "decision": tdecision,
            "orig_inp": tfs_input,
            "cover_inp": tft_input,
        }
