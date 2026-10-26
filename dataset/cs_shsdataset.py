from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import audio_conv as ac
import dataset.data_augmentation as da


class SHSDataset(Dataset):
    """_summary_: DataLoader for Pair SHS100K Dataset
        pair_{split}.csv contains Performances(original song) and Cliques(cover song)
        [file structure]
        data
        ㄴshs-100k
            ㄴpair_train (1703 pairs | 3406 songs)
                ㄴwork_id{i}.mp3
                    ㄴperf_id1.mp3
                    ㄴperf_id2.mp3
            ㄴpair_validate (73 pairs | 146 songs)
            ㄴpair_test (111 pairs | 222 songs)
    Attr:
        pair_csv_path: metadata path of csv file
        dpath: data path - path to data / shs-100k
        inp_rep: input representation
        crop_len: length of input time frame
        split: train, validate, test division
        seed: set seed for random generator for train and validate dataset
        cache_rep: whether or not to cash/use cached input feature
        cache_dir: directory of cache
    """
    def __init__(self, pair_csv_path, dpath, inp_rep, crop_len, split, seed=2026, cache_rep=True, cache_dir=None):
        # initialize attribute
        self.pair_csv_path = Path(dpath) / pair_csv_path # only start part should be path
        self.dpath = Path(dpath) / "_".join(["pair", split])
        self.inp_rep = inp_rep
        self.crop_len = crop_len
        self.split = split # train, val, test
        self.src_paths = [] # paths
        self.tgt_paths = []
        self.neg_tgt_paths = []
        self.src_work_ids = [] # work_ids
        self.tgt_work_ids = [] 
        self.neg_tgt_work_ids = []
        self.src_performance_ids = [] # performance_ids
        self.tgt_performance_ids = []
        self.neg_tgt_performance_ids = []
        self.random_gen = random.Random(seed)
        self.cache_rep = cache_rep

        if cache_dir is None:
            if self.inp_rep.lower() == "cqt":
                self.cache_dir = Path("cache") / "cqt_cache" / "_".join([self.dpath.name, inp_rep.lower()]) # default cache_dir
            elif self.inp_rep.lower() == "hpcp":
                self.cache_dir = Path("cache") / "hpcp_cache" / "_".join([self.dpath.name, inp_rep.lower()]) # default cache_dir
        else:
            self.cache_dir = Path(cache_dir)

        if self.cache_rep:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # read csv
        cnames = ["performance_id", "work_id", "performing_title", "performing_artist", "youtube_id"]
        pair_df = pd.read_csv(self.pair_csv_path, header=None, names=cnames)
        pair_df["performance_id"] = (pair_df["performance_id"].astype(str)) # convert performance_id into string
        pair_df["work_id"] = (pair_df["work_id"].astype(str)) # convert work_id into string
        
        assert len(pair_df) % 2 == 0, ("should have even number of music")

        # load all source-target file (dir/fname.mp3) in order
        for pos in range(0, len(pair_df), 2):
            src_row = pair_df.iloc[pos]
            tgt_row = pair_df.iloc[pos + 1]

            assert pair_df.duplicated().sum() == 0
            assert src_row["work_id"] == tgt_row["work_id"]
            work_id = src_row["work_id"]

            candidate_indices = pair_df.index[pair_df.iloc[:, 1] != work_id].tolist()

            neg_pos = self.random_gen.choice(candidate_indices)
            neg_row = pair_df.iloc[neg_pos]

            src_work_id = str(src_row["work_id"])
            tgt_work_id = str(tgt_row["work_id"])
            neg_tgt_work_id = str(neg_row["work_id"])

            self.src_work_ids.append(src_work_id)
            self.tgt_work_ids.append(tgt_work_id)
            self.neg_tgt_work_ids.append(neg_tgt_work_id)

            src_pid = str(src_row["performance_id"])
            tgt_pid = str(tgt_row["performance_id"])
            neg_tgt_pid = str(neg_row["performance_id"])

            self.src_performance_ids.append(src_pid) # save performance id
            self.tgt_performance_ids.append(tgt_pid)
            self.neg_tgt_performance_ids.append(neg_tgt_pid)

            src_path = self.dpath / src_work_id / "{}.mp3".format(src_pid)
            tgt_path = self.dpath / tgt_work_id / "{}.mp3".format(tgt_pid)
            neg_tgt_path = self.dpath / neg_tgt_work_id / "{}.mp3".format(neg_tgt_pid)

            self.src_paths.append(src_path)
            self.tgt_paths.append(tgt_path)
            self.neg_tgt_paths.append(neg_tgt_path)

        self.random_gen = random.Random(seed)

        pairs = list(zip(self.src_paths, self.tgt_paths, self.neg_tgt_paths,
                    self.src_work_ids, self.tgt_work_ids, self.neg_tgt_work_ids,
                    self.src_performance_ids, self.tgt_performance_ids, self.neg_tgt_performance_ids))

        self.random_gen.shuffle(pairs) # shuffle in pair
        self.src_paths, self.tgt_paths, self.neg_tgt_paths, self.src_work_ids, self.tgt_work_ids, self.neg_tgt_work_ids, self.src_performance_ids, self.tgt_performance_ids, self.neg_tgt_performance_ids = map(list, zip(*pairs)) # reassign

    def __len__(self):
        return len(self.src_performance_ids)

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
        else: # Nnt cached yet
            feature = get_feature(feature_path)
            feature = torch.from_numpy(np.asarray(feature, dtype=np.float32))
            torch.save(feature, cache_path)

            return feature
        
    
    def __getitem__(self, pidx):
        # pair-level metadata: length = 3406
        src_wid = self.src_work_ids[pidx]
        tgt_wid = self.tgt_work_ids[pidx]
        neg_tgt_wid = self.neg_tgt_work_ids[pidx]

        src_pid = str(self.src_performance_ids[pidx])
        tgt_pid = str(self.tgt_performance_ids[pidx])
        neg_tgt_pid = str(self.neg_tgt_performance_ids[pidx])

        # Find actual audio paths using PID
        src_path = self.src_paths[pidx]
        tgt_path = self.tgt_paths[pidx]
        neg_tgt_path = self.neg_tgt_paths[pidx]

        src_feature = self.get_feature(src_path, src_pid)
        tgt_feature = self.get_feature(tgt_path, tgt_pid)
        neg_tgt_feature = self.get_feature(neg_tgt_path, neg_tgt_pid)

        fs_input = self.test_crop_1(src_feature)
        ft_input = self.test_crop_1(tgt_feature)
        fn_input = self.test_crop_1(neg_tgt_feature)
        
        fs_input = da.mirror_pitch_axis(fs_input)
        ft_input = da.mirror_pitch_axis(ft_input)
        fn_input = da.mirror_pitch_axis(fn_input)
        # data augmentation on source only
        # fs_input = da.shift_pitch_cqt(fs_input)
        # fs_input = da.shift_mask_cqt_time(fs_input)
        # fs_input = da.partial_cqt(fs_input, p=0.5)

        tfs_input = torch.as_tensor(fs_input, dtype=torch.float32).unsqueeze(0)
        tft_input = torch.as_tensor(ft_input, dtype=torch.float32).unsqueeze(0)
        tfnt_input = torch.as_tensor(fn_input, dtype=torch.float32).unsqueeze(0)

        return {
            "src_work_id": int(src_wid),
            "pos_work_id": int(tgt_wid),
            "neg_work_id": int(neg_tgt_wid),
            "src_label": int(src_pid),
            "pos_label": int(tgt_pid),
            "neg_label": int(neg_tgt_pid),
            "orig_inp": tfs_input,
            "pos_inp": tft_input,
            "neg_inp": tfnt_input
        }
