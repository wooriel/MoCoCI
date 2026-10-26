from pathlib import Path
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import dataset.data_augmentation as da


class PNDATACOSDataset(Dataset):
    """_summary_: DataLoader for Pair DATACOS Dataset
        da_tacos/da-tacos_metadata/da-tacos_benchmark_subset_metadata.json contains relation of Original Work and Performance(cover song)
        [file structure]
        data
        ㄴda_tacos
            ㄴda-tacos_benchmark_subset_hpcp
                ㄴW_{work_id}_hpcp
                    ㄴP_{perf_id}.mp3
                    ㄴP_{perf_id2}.mp3
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
    def __init__(self, csv_path, dpath, inp_rep, crop_len, split, seed=None, cache_dir=None):
        # initialize attribute
        self.csv_path = Path(csv_path) # only start part should be path
        self.dpath = Path(dpath) # / "_".join(["pair", split])
        self.inp_rep = inp_rep
        self.crop_len = crop_len
        self.split = split # train, val, test

        # self.unique_work_ids = [] # stores work_id of all individual song
        # self.unique_perf_ids = [] # stores perf_id of all individual song
        # self.unique_paths = [] # stores unique path of positive path

        self.src_work_ids = [] # work_id for song 1
        self.tgt_work_ids = [] # work_id for song 2
        self.neg_tgt_work_ids = [] # work_id for song 3

        self.src_performance_ids = [] # performance_id for song 1
        self.tgt_performance_ids = [] # performance_id for song 2
        self.neg_tgt_performance_ids = [] # performance_id for song 3

        self.src_paths = []
        self.tgt_paths = []
        self.neg_tgt_paths = []

        self.random_gen = random.Random(seed)
        
        if cache_dir is None:
            self.cache_dir = Path("cache") / "hpcp_cache" / self.dpath.name # default cache_dir
        else:
            self.cache_dir = Path(cache_dir)

        # self.cache_dir.mkdir(parents=True, exist_ok=True)
        # should update src_paths, tgt_paths, src_perf_ids, tgt_perf_ids, decisions
        pair_df = pd.read_csv(self.csv_path, header=None, names=["src_work", "src_perf", "pos_tgt_work", "pos_tgt_perf", "pos_decisions", "neg_tgt_work", "neg_tgt_perf", "neg_decisions"])
        self.src_work_ids = pair_df["src_work"].astype(str).tolist()
        self.tgt_work_ids = pair_df["pos_tgt_work"].astype(str).tolist()
        self.neg_tgt_work_ids = pair_df["neg_tgt_work"].astype(str).tolist()
        self.src_performance_ids = pair_df["src_perf"].astype(str).tolist()
        self.tgt_performance_ids = pair_df["pos_tgt_perf"].astype(str).tolist()
        self.neg_tgt_performance_ids = pair_df["neg_tgt_perf"].astype(str).tolist()
        self.src_paths = [self.cache_dir / "{}.pt".format(spid) for spid in self.src_performance_ids]
        self.tgt_paths = [self.cache_dir / "{}.pt".format(spid) for spid in self.tgt_performance_ids]
        self.neg_tgt_paths = [self.cache_dir / "{}.pt".format(spid) for spid in self.neg_tgt_performance_ids]

        all_perf_ids = (
            pair_df["src_perf"].astype(str).tolist()
            + pair_df["pos_tgt_perf"].astype(str).tolist()
            + pair_df["neg_tgt_perf"].astype(str).tolist()
        )

        # self.unique_perf_ids = list(dict.fromkeys(all_perf_ids)) # unique id
        # self.unique_paths = [self.cache_dir / "{}.pt".format(pid) for pid in self.unique_perf_ids]
        
        # # after appending single path / perf_id to unique path and id, make dictionary
        # self.unique_perf_id_to_path = {
        #     str(pid): path
        #     for pid, path in zip(
        #         self.unique_perf_ids,
        #         self.unique_paths
        #     )
        # }
        
        self.random_gen = random.Random(seed)
        pairs = list(zip(self.src_paths, self.tgt_paths, self.neg_tgt_paths,
                self.src_work_ids, self.tgt_work_ids, self.neg_tgt_work_ids,
                self.src_performance_ids, self.tgt_performance_ids, self.neg_tgt_performance_ids))
        if len(pairs) == 0:
            raise ValueError("Pair CSV is empty")
        
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

    
    def get_feature(self, feature_path):
        if not feature_path.exists():
            raise FileNotFoundError("Cached feature not found: {}".format(feature_path))

        return torch.load(
            feature_path,
            map_location="cpu",
            weights_only=True,
        )
    
    def expand_hpcp_vertical(self, x, target_bins=96):
        current_bins = x.shape[-2]

        assert target_bins % current_bins == 0

        repeat_factor = target_bins // current_bins

        return x.repeat_interleave(repeat_factor, dim=-2)
    
    def __getitem__(self, pidx):
        # pair-level metadata: length = 3406
        src_wid = str(self.src_work_ids[pidx])
        pos_wid = str(self.tgt_work_ids[pidx])
        neg_wid = str(self.neg_tgt_work_ids[pidx])

        src_pid = str(self.src_performance_ids[pidx])
        pos_pid = str(self.tgt_performance_ids[pidx])
        neg_pid = str(self.neg_tgt_performance_ids[pidx])

        src_path = self.src_paths[pidx]
        pos_path = self.tgt_paths[pidx]
        neg_path = self.neg_tgt_paths[pidx]

        src_feature = self.get_feature(src_path)
        pos_feature = self.get_feature(pos_path)
        neg_feature = self.get_feature(neg_path)

        fs_input = self.test_crop_1(src_feature)
        fp_input = self.test_crop_1(pos_feature)
        fn_input = self.test_crop_1(neg_feature)

        # augmentation during training
        if self.split == "train":
            fs_input = da.shift_pitch_hpcp(fs_input)
            fs_input = da.mirror_pitch_axis(fs_input)
            fs_input = da.shift_mask_hpcp_time(fs_input)
            fs_input = da.partial_gap_hpcp(fs_input)
            # fs_input = da.splice_same_song_hpcp(fs_input, p=0.2, seg_len=500)

            # fp_input = da.augment_hpcp(fp_input)
            # fp_input = da.splice_same_song_hpcp(fp_input, p=0.2, seg_len=500)

            # fn_input = da.augment_hpcp(fn_input)
            # fn_input = da.splice_same_song_hpcp(fn_input, p=0.2, seg_len=500)
        else:
            fs_input = da.mirror_pitch_axis(fs_input)
        fp_input = da.mirror_pitch_axis(fp_input)
        fn_input = da.mirror_pitch_axis(fn_input)
        tfs_input = torch.as_tensor(fs_input, dtype=torch.float32).unsqueeze(0)
        tfp_input = torch.as_tensor(fp_input, dtype=torch.float32).unsqueeze(0)
        tfn_input = torch.as_tensor(fn_input, dtype=torch.float32).unsqueeze(0)

        # tfs_input = self.expand_hpcp_vertical(tfs_input, target_bins=96)
        # tfp_input = self.expand_hpcp_vertical(tfp_input, target_bins=96)
        # tfn_input = self.expand_hpcp_vertical(tfn_input, target_bins=96)

        return {
            "src_work_id": int(src_wid),
            "pos_work_id": int(pos_wid),
            "neg_work_id": int(neg_wid),
            "src_label": int(src_pid),
            "pos_label": int(pos_pid),
            "neg_label": int(neg_pid),
            "orig_inp": tfs_input,
            "pos_inp": tfp_input,
            "neg_inp": tfn_input
        }
