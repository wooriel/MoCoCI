from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import audio_conv as ac
import dataset.data_augmentation as da


class Cover80Dataset(Dataset):
    """_summary_: DataLoader for Cover80
        list1.list contains original work and list2.list contains cover song
        [file structure]
        data
        ㄴcovers80
            ㄴlist1.list
            ㄴlist2.list
            ㄴdirectories of song
    Attr:
        lpath_src: metadata path of original music
        lpath_tgt: metadata path of coversong music
        dpath: data path - path to data / cover80
        inp_rep: input representation
        crop_len: length of input time frame
        cache_rep: whether or not to cash/use cached input feature
        cache_dir: directory of cache
    """
    def __init__(self, lpath_src, lpath_tgt, dpath, inp_rep, crop_len, seed, cache_rep=True, cache_dir=None):
        self.lpath_src = Path(dpath) / lpath_src # list1.txt 위치 (csv파일과 유사)
        self.lpath_tgt = Path(dpath) / lpath_tgt # list2.txt 위치 (csv파일과 유사)
        self.dpath = Path(dpath)
        self.inp_rep = inp_rep
        self.crop_len = crop_len

        self.src_work_ids = []
        self.tgt_work_ids = []
        self.neg_tgt_work_ids = []

        self.src_perf_ids = []
        self.tgt_perf_ids = []
        self.neg_tgt_perf_ids = []
        
        self.src_paths = []
        self.tgt_paths = []
        self.neg_tgt_paths = []
        self.cache_rep = cache_rep
        self.random_gen = random.Random(seed)

        if cache_dir is None:
            self.cache_dir = Path("cache") / "{}_cache".format(self.inp_rep.lower()) / self.dpath.name # default cache_dir
        else:
            self.cache_dir = Path(cache_dir)

        if self.cache_rep:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # load all source-target file (dir/fname.mp3) in .list file in order
        with open(self.lpath_src, "r") as f:
            src_paths = [self.dpath / "".join([line.strip(), ".mp3"]) for line in f if line.strip()] # filter empty space

        with open(self.lpath_tgt, "r") as f:
            tgt_paths = [self.dpath / "".join([line.strip(), ".mp3"]) for line in f if line.strip()]

        assert len(src_paths) == len(tgt_paths), "pair length not match"
        self.src_paths = src_paths
        self.tgt_paths = tgt_paths

        self.src_idx_to_path = {
            i: path
            for i, path in enumerate(self.src_paths)
        }

        self.tgt_idx_to_path = {
            j: path
            for j, path in enumerate(self.tgt_paths)
        }

        for i in range(len(self.src_paths)):
            self.src_work_ids.append(i)
            self.tgt_work_ids.append(i)
            self.src_perf_ids.append(i * 2)
            self.tgt_perf_ids.append(i * 2 + 1)

        self.random_gen = random.Random(seed)
        
        wlen = len(self.tgt_work_ids)
        for i in range(wlen):
            candidate_indices = [
                j for j in range(wlen)
                if j != i
            ]

            neg_tgt_idx = self.random_gen.sample(candidate_indices, k=1)[0]
            neg_tgt_work_id = self.tgt_work_ids[neg_tgt_idx]
            self.neg_tgt_work_ids.append(neg_tgt_work_id)
            neg_tgt_perf_id = self.tgt_perf_ids[neg_tgt_idx]
            self.neg_tgt_perf_ids.append(neg_tgt_perf_id)
            neg_tgt_path = self.tgt_paths[neg_tgt_idx]
            self.neg_tgt_paths.append(neg_tgt_path)

        pairs = list(zip(self.src_paths, self.tgt_paths, self.neg_tgt_paths,
                self.src_work_ids, self.tgt_work_ids, self.neg_tgt_work_ids,
                self.src_perf_ids, self.tgt_perf_ids, self.neg_tgt_perf_ids))

        self.random_gen.shuffle(pairs) # shuffle in pair
        self.src_paths, self.tgt_paths, self.neg_tgt_paths, self.src_work_ids, self.tgt_work_ids, self.neg_tgt_work_ids, self.src_perf_ids, self.tgt_perf_ids, self.neg_tgt_perf_ids = map(list, zip(*pairs)) # reassign

    def __len__(self):
        return len(self.src_perf_ids)
    

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
        if self.inp_rep == "CQT":
            get_feature = ac.get_cqt_spec
        elif self.inp_rep == "HPCP":
            get_feature = ac.get_hpcp
        else:
            raise ValueError("Unknown input representation: {}".format(self.inp_rep))

        if not self.cache_rep: # have to calculate
            feature = get_feature(feature_path)
            return torch.from_numpy(feature).float()

        cache_path = self.cache_dir / "{}.pt".format(feature_path.stem)

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
        src_wid = str(self.src_work_ids[pidx])
        pos_wid = str(self.tgt_work_ids[pidx])
        neg_wid = str(self.neg_tgt_work_ids[pidx])

        src_pid = str(self.src_perf_ids[pidx])
        pos_pid = str(self.tgt_perf_ids[pidx])
        neg_pid = str(self.neg_tgt_perf_ids[pidx])

        src_path = self.src_paths[pidx]
        pos_path = self.tgt_paths[pidx]
        neg_path = self.neg_tgt_paths[pidx]

        src_feature = self.get_feature(src_path)
        pos_feature = self.get_feature(pos_path)
        neg_feature = self.get_feature(neg_path)

        fs_input = self.test_crop_1(src_feature)
        fp_input = self.test_crop_1(pos_feature)
        fn_input = self.test_crop_1(neg_feature)

        fs_input = da.mirror_pitch_axis(fs_input)
        fp_input = da.mirror_pitch_axis(fp_input)
        fn_input = da.mirror_pitch_axis(fn_input)

        tfs_input = torch.as_tensor(fs_input, dtype=torch.float32).unsqueeze(0)
        tfp_input = torch.as_tensor(fp_input, dtype=torch.float32).unsqueeze(0)
        tfn_input = torch.as_tensor(fn_input, dtype=torch.float32).unsqueeze(0)

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
