from pathlib import Path
import random
from itertools import combinations
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import audio_conv as ac


class PREPNDATACOSDataset(Dataset):
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
    def __init__(self, csv_path, dpath, inp_rep, crop_len, split, seed=None, cache_rep=True, cache_dir=None):
        # initialize attribute
        self.csv_path = Path(csv_path) # only start part should be path
        self.dpath = Path(dpath) # / "_".join(["pair", split])
        self.inp_rep = inp_rep
        self.crop_len = crop_len
        self.split = split # train, val, test
        self.unique_work_ids = [] # stores work_id of all individual song
        self.unique_perf_ids = [] # stores perf_id of all individual song
        self.unique_paths = [] # stores unique path of positive path

        self.single_song = []
        self.ppair_song = []

        self.src_work_id = [] # work_id for song 1
        self.tgt_work_id = [] # work_id for song 2
        self.neg_tgt_work_id = [] # work_id for song 3

        self.src_performance_ids = [] # performance_id for song 1
        self.tgt_performance_ids = [] # performance_id for song 2
        self.neg_tgt_performance_ids = [] # performance_id for song 3

        self.src_paths = []
        self.tgt_paths = []
        self.neg_tgt_paths = []
        
        self.random_gen = random.Random(seed)
        self.cache_rep = cache_rep

        if cache_dir is None:
            self.cache_dir = Path("cache") / "hpcp_cache" / self.dpath.name # default cache_dir
        else:
            self.cache_dir = Path(cache_dir)

        if self.cache_rep:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        csv_flag = False
        csv_train_path = self.csv_path / "tri_train.csv"
        if not csv_train_path.exists():
            csv_flag = True
            self.csv_path.mkdir(parents=True, exist_ok=True)
            

        csv_validate_path = self.csv_path / "tri_validate.csv"
        if not csv_validate_path.exists():
            self.csv_path.mkdir(parents=True, exist_ok=True)
            

        csv_test_path = self.csv_path / "tri_test.csv"
        if not csv_test_path.exists():
            self.csv_path.mkdir(parents=True, exist_ok=True)
            
        for wdir in self.dpath.iterdir():
            if not wdir.is_dir():
                continue

            works = []
            work_id = wdir.name.split("_")[1] # work id in string
                
            for file_path in wdir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".h5":
                    perf_id = file_path.name.split("_")[1]
                    works.append((work_id, perf_id, file_path))

            if len(works) == 1: # add to negative pair
                self.single_song.extend(works)
            elif len(works) > 1: # add to positive pairs
                self.ppair_song.append(works) # this is nested positive pairs [[perf. within same work 1], [perf. within same work 2]]

        # update unique element (work_id, perf_id, path)
        self.unique_work_ids.extend(perf[0] for works in self.ppair_song for perf in works)
        self.unique_perf_ids.extend(perf[1] for works in self.ppair_song for perf in works)
        self.unique_paths.extend(perf[2] for works in self.ppair_song for perf in works)
        self.unique_work_ids.extend(x[0] for x in self.single_song)
        self.unique_perf_ids.extend(x[1] for x in self.single_song)
        self.unique_paths.extend(x[2] for x in self.single_song)

        # split into 8:1:1, index of the song
        total_pos = len(self.ppair_song)
        total_neg = len(self.single_song)
        ptrain_pos = int(total_pos * 0.8)
        pval_pos = int(total_pos * 0.1)
        ntrain_neg = int(total_neg * 0.8)
        nval_neg = int(total_neg * 0.1)

        # split sample
        pos_train_samp = self.ppair_song[:ptrain_pos]
        pos_val_samp = self.ppair_song[ptrain_pos:ptrain_pos + pval_pos]
        pos_test_samp = self.ppair_song[ptrain_pos + pval_pos:]

        neg_train_samp = self.single_song[:ntrain_neg]
        neg_val_samp = self.single_song[ntrain_neg:ntrain_neg + nval_neg]
        neg_test_samp = self.single_song[ntrain_neg + nval_neg:]

        # pos_train/val/test_samp is nested!
        train_pool = [perf for works in pos_train_samp for perf in works] + neg_train_samp 
        val_pool = [perf for works in pos_val_samp for perf in works] + neg_val_samp
        test_pool = [perf for works in pos_test_samp for perf in works] + neg_test_samp

        # positive combinations
        pos_train_comb_list, pos_val_comb_list, pos_test_comb_list = [], [], []
        for works in pos_train_samp: # nested list
            for a, p in combinations(works, 2):
                pos_train_comb_list.append((a, p))
                pos_train_comb_list.append((p, a))
        for works in pos_val_samp:
            for a, p in combinations(works, 2):
                pos_val_comb_list.append((a, p))
                pos_val_comb_list.append((p, a))
        for works in pos_test_samp:
            for a, p in combinations(works, 2):
                pos_test_comb_list.append((a, p))
                pos_test_comb_list.append((p, a))

        # add negative pair to positive pair
        train_comb_list, val_comb_list, test_comb_list = [], [], []
        for pos_pair in pos_train_comb_list:
            pos_work_id = pos_pair[0][0] # work_id of src elem
            neg_elem = self.random_gen.choice(train_pool)
            while pos_work_id == neg_elem[0]:
                neg_elem = self.random_gen.choice(train_pool)
            train_comb_list.append((pos_pair[0], pos_pair[1], neg_elem)) # index of negative element

        for pos_pair in pos_val_comb_list:
            pos_work_id = pos_pair[0][0] # work_id of src elem
            neg_elem = self.random_gen.choice(val_pool)
            while pos_work_id == neg_elem[0]:
                neg_elem = self.random_gen.choice(val_pool)
            val_comb_list.append((pos_pair[0], pos_pair[1], neg_elem)) # index of negative element

        for pos_pair in pos_test_comb_list:
            pos_work_id = pos_pair[0][0] # work_id of src elem
            neg_elem = self.random_gen.choice(test_pool)
            while pos_work_id == neg_elem[0]:
                neg_elem = self.random_gen.choice(test_pool)
            test_comb_list.append((pos_pair[0], pos_pair[1], neg_elem)) # index of negative element

        train_src_work_ids, train_tgt_work_ids, neg_train_tgt_work_ids = [], [], []
        train_src_performance_ids, train_tgt_performance_ids, neg_train_tgt_performance_ids = [], [], []
        train_src_paths, train_tgt_paths, neg_train_tgt_paths = [], [], []
        for train_pair in train_comb_list:
            train_src_work_ids.append(train_pair[0][0])
            train_tgt_work_ids.append(train_pair[1][0])
            neg_train_tgt_work_ids.append(train_pair[2][0])
            train_src_performance_ids.append(train_pair[0][1])
            train_tgt_performance_ids.append(train_pair[1][1])
            neg_train_tgt_performance_ids.append(train_pair[2][1])
            train_src_paths.append(train_pair[0][2])
            train_tgt_paths.append(train_pair[1][2])
            neg_train_tgt_paths.append(train_pair[2][2])
        self.src_work_id.extend(train_src_work_ids) # extend work_ids
        self.tgt_work_id.extend(train_tgt_work_ids)
        self.neg_tgt_work_id.extend(neg_train_tgt_work_ids)
        self.src_performance_ids.extend(train_src_performance_ids) # extend perf_ids
        self.tgt_performance_ids.extend(train_tgt_performance_ids)
        self.neg_tgt_performance_ids.extend(neg_train_tgt_performance_ids)
        self.src_paths.extend(train_src_paths) # extend to paths
        self.tgt_paths.extend(train_tgt_paths)
        self.neg_tgt_paths.extend(neg_train_tgt_paths)

        val_src_work_ids, val_tgt_work_ids, neg_val_tgt_work_ids = [], [], []
        val_src_performance_ids, val_tgt_performance_ids, neg_val_tgt_performance_ids = [], [], []
        val_src_paths, val_tgt_paths, neg_val_tgt_paths = [], [], []
        for val_pair in val_comb_list:
            val_src_work_ids.append(val_pair[0][0])
            val_tgt_work_ids.append(val_pair[1][0])
            neg_val_tgt_work_ids.append(val_pair[2][0])
            val_src_performance_ids.append(val_pair[0][1])
            val_tgt_performance_ids.append(val_pair[1][1])
            neg_val_tgt_performance_ids.append(val_pair[2][1])
            val_src_paths.append(val_pair[0][2])
            val_tgt_paths.append(val_pair[1][2])
            neg_val_tgt_paths.append(val_pair[2][2])
        self.src_work_id.extend(val_src_work_ids) # extend work_ids
        self.tgt_work_id.extend(val_tgt_work_ids)
        self.neg_tgt_work_id.extend(neg_val_tgt_work_ids)
        self.src_performance_ids.extend(val_src_performance_ids) # extend perf_ids
        self.tgt_performance_ids.extend(val_tgt_performance_ids)
        self.neg_tgt_performance_ids.extend(neg_val_tgt_performance_ids)
        self.src_paths.extend(val_src_paths) # extend to paths
        self.tgt_paths.extend(val_tgt_paths)
        self.neg_tgt_paths.extend(neg_val_tgt_paths)

        test_src_work_ids, test_tgt_work_ids, neg_test_tgt_work_ids = [], [], []
        test_src_performance_ids, test_tgt_performance_ids, neg_test_tgt_performance_ids = [], [], []
        test_src_paths, test_tgt_paths, neg_test_tgt_paths = [], [], []
        for test_pair in test_comb_list:
            test_src_work_ids.append(test_pair[0][0])
            test_tgt_work_ids.append(test_pair[1][0])
            neg_test_tgt_work_ids.append(test_pair[2][0])
            test_src_performance_ids.append(test_pair[0][1])
            test_tgt_performance_ids.append(test_pair[1][1])
            neg_test_tgt_performance_ids.append(test_pair[2][1])
            test_src_paths.append(test_pair[0][2])
            test_tgt_paths.append(test_pair[1][2])
            neg_test_tgt_paths.append(test_pair[2][2])
        self.src_work_id.extend(test_src_work_ids) # extend work_ids
        self.tgt_work_id.extend(test_tgt_work_ids)
        self.neg_tgt_work_id.extend(neg_test_tgt_work_ids)
        self.src_performance_ids.extend(test_src_performance_ids) # extend perf_ids
        self.tgt_performance_ids.extend(test_tgt_performance_ids)
        self.neg_tgt_performance_ids.extend(neg_test_tgt_performance_ids)
        self.src_paths.extend(test_src_paths) # extend to paths
        self.tgt_paths.extend(test_tgt_paths)
        self.neg_tgt_paths.extend(neg_test_tgt_paths)

        pair_len = len(train_comb_list) + len(val_comb_list) + len(test_comb_list)
        # assume that 1st elem: anchor | 2nd elem: pos pair | 3rd elem: neg pair

        if csv_flag:
            ac.write_csv(csv_train_path, zip(train_src_work_ids, train_src_performance_ids,
                train_tgt_work_ids, train_tgt_performance_ids, [1 for i in range(len(train_comb_list))],
                neg_train_tgt_work_ids, neg_train_tgt_performance_ids, [0 for i in range(len(train_comb_list))]
            ))
            ac.write_csv(csv_validate_path, zip(val_src_work_ids, val_src_performance_ids,
                val_tgt_work_ids, val_tgt_performance_ids, [1 for i in range(len(val_comb_list))],
                neg_val_tgt_work_ids, neg_val_tgt_performance_ids, [0 for i in range(len(val_comb_list))]
            ))
            ac.write_csv(csv_test_path, zip(test_src_work_ids, test_src_performance_ids,
                test_tgt_work_ids, test_tgt_performance_ids, [1 for i in range(len(test_comb_list))],
                neg_test_tgt_work_ids, neg_test_tgt_performance_ids, [0 for i in range(len(test_comb_list))]
            ))
            
        assert len(self.src_performance_ids) == len(self.tgt_performance_ids) == len(self.neg_tgt_performance_ids)
        assert len(self.src_performance_ids) == len(self.src_paths) == len(self.tgt_paths) == len(self.neg_tgt_paths)
        assert len(train_src_performance_ids) == len(pos_train_comb_list)

        # after appending single path / perf_id to unique path and id, make dictionary
        self.unique_perf_id_to_path = {
            str(pid): path
            for pid, path in zip(
                self.unique_perf_ids,
                self.unique_paths
            )
        }

        pairs = list(zip(self.src_paths, self.tgt_paths, self.neg_tgt_paths,
                         self.src_work_id, self.tgt_work_id, self.neg_tgt_work_id,
                         self.src_performance_ids, self.tgt_performance_ids, self.neg_tgt_performance_ids))
        self.random_gen.shuffle(pairs) # shuffle in pair
        self.src_paths, self.tgt_paths, self.neg_tgt_paths, self.src_work_id, self.tgt_work_id, self.neg_tgt_work_id, self.src_performance_ids, self.tgt_performance_ids, self.neg_tgt_performance_ids = map(list, zip(*pairs)) # reassign

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
        if self.inp_rep == "HPCP":
            get_feature = ac.load_single_h5py
        else:
            raise ValueError("Unknown input representation: {}".format(self.inp_rep))

        cache_path = self.cache_dir / "{}.pt".format(performance_id)
        if cache_path.exists():
            return torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )
        else: # Nnt cached yet
            feature = get_feature(feature_path) # load_single_h5py
            feature = torch.from_numpy(np.asarray(feature, dtype=np.float32))
            torch.save(feature, cache_path)

            return feature
    
    def __getitem__(self, pidx):
        src_wid = str(self.src_work_id[pidx])
        pos_wid = str(self.tgt_work_id[pidx])
        neg_wid = str(self.neg_tgt_work_id[pidx])

        src_pid = str(self.src_performance_ids[pidx])
        pos_pid = str(self.tgt_performance_ids[pidx])
        neg_pid = str(self.neg_tgt_performance_ids[pidx])

        src_path = self.src_paths[pidx]
        pos_path = self.tgt_paths[pidx]
        neg_path = self.neg_tgt_paths[pidx]

        src_feature = self.get_feature(src_path, src_pid)
        pos_feature = self.get_feature(pos_path, pos_pid)
        neg_feature = self.get_feature(neg_path, neg_pid)

        fs_input = self.test_crop_1(src_feature)
        fp_input = self.test_crop_1(pos_feature)
        fn_input = self.test_crop_1(neg_feature)

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