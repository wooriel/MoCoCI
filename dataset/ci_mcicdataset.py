from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import audio_conv as ac
import dataset.data_augmentation as da


class MCICDataset(Dataset):
    """_summary_: DataLoader for Pair MCIC Dataset
        MCIC.csv contains Complaining work(source song) and Defending work(target song)
        [file structure]
        data
        ㄴMCIC
            ㄴMCIC DATA
                ㄴMIDI FILES .mid (116 pairs | 232 songs)
                    ㄴ001_a_I_hear_... (File ID)
                    ㄴ001_b_...
                ㄴSCORE FILES .pdf (116 pairs | 232 songs)
        
    Attr:
        pair_csv_path: metadata path of csv file
        dpath: data path - path to data / MCIC
        inp_rep: input representation
        crop_len: length of input time frame
        split: train, validate, test division
        cache_rep: whether or not to cash/use cached input feature
        cache_dir: directory of cache
    """
    def __init__(self, pair_csv_path, dpath, inp_rep, crop_len, cache_rep=True, cache_dir=None, crop=False):
        # initialize attribute
        self.pair_csv_path = Path(dpath) / pair_csv_path # only start part should be path
        self.dpath = Path(dpath)
        self.inp_rep = inp_rep
        self.crop_len = crop_len
        self.src_paths = []
        self.tgt_paths = []
        self.work_ids = [] # label for pair
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
        copy_df = copy_df.dropna(how="all").reset_index(drop=True) # remove empty rows
        
        assert len(copy_df) % 2 == 0, ("should have even number of music")

        file_dictionary = {
            self.clean_file_id(file_path.name): file_path
            for file_path in self.dpath.iterdir()
            if file_path.is_file()
        }

        # load all source-target file (dir/fname.mp3) in order
        for pos in range(0, len(copy_df), 2):
            src_row = copy_df.iloc[pos]
            tgt_row = copy_df.iloc[pos + 1]

            decision = str(src_row["Decision"])
            if decision == "-": # do not use settled case in dataset
                continue
            self.decisions.append(int(decision))

            work_id = int(str(src_row["Case"]).split(" ")[1]) # "Case #" -> #
            tgt_work_id = int(str(src_row["Case"]).split(" ")[1]) # target work_id
            assert work_id == tgt_work_id, ("should have same id")
            self.work_ids.append(work_id)
            self.src_perf_ids.append(work_id * 2)
            self.tgt_perf_ids.append(work_id * 2 + 1)

            src_name = self.clean_file_id(str(src_row["File ID"]))
            tgt_name = self.clean_file_id(str(tgt_row["File ID"]))

            src_path = file_dictionary[src_name]
            tgt_path = file_dictionary[tgt_name]

            self.src_paths.append(src_path)
            self.tgt_paths.append(tgt_path)


    def __len__(self):
        return len(self.src_paths)


    def clean_file_id(self, file_id):
        # return str(file_id).replace(",", "").replace(" ", "").replace("’", "").replace("\"", "").replace("“", "").replace("”", "")
        pref = re.sub(r"[, ()'\'\"\.\;?!\[\]‘’“”]", '', str("".join(file_id.split(".")[:-1])))
        cleaned_fname = ".".join([pref, "mid"])

        return cleaned_fname


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
            feature = get_feature(feature_path, format="mid")
            return torch.from_numpy(feature).float()

        cache_path = self.cache_dir / "{}.pt".format(performance_id)

        if cache_path.exists():
            return torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )
        else: # Nnt cached yet
            feature = get_feature(feature_path, format="mid")
            feature = torch.from_numpy(np.asarray(feature, dtype=np.float32))
            torch.save(feature, cache_path)

            return feature


    def __getitem__(self, pidx):
        src_path, tgt_path, = self.src_paths[pidx], self.tgt_paths[pidx]
        work_id = self.work_ids[pidx]
        decision = self.decisions[pidx]
        src_pid, tgt_pid = self.src_perf_ids[pidx], self.tgt_perf_ids[pidx]
        src_cid, tgt_cid = "{:03d}a".format(work_id), "{:03d}b".format(work_id)

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
            "work_id": twork_id,
            "src_perf_id": tsrc_pid,
            "tgt_perf_id": ttgt_pid,
            "decision": tdecision,
            "orig_inp": tfs_input,
            "cover_inp": tft_input,
        }
