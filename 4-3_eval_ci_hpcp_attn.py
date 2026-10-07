import os
import argparse
from pathlib import Path
import torch
from dataset import cs_cover80dataset, cs_shsdataset, ci_copyright40dataset, ci_mcicdataset, pndatacosdataset
from torch.utils.data import DataLoader
from models import a_vittokenizer, b_selfattention, c_self_sim_model
from ci_threshold import calculate_ci_metric, log_ci_eval
from checkpoint import load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument("--csv_pref", type=str, default="pair")
parser.add_argument("--exp_name", type=str, default="hpcp_attn_v13", help="name of training experiment")
parser.add_argument("--cache_dir", type=str, default="cache", help="cache directory name")
parser.add_argument("--rep", type=str, default="hpcp", help="representation of the input")
parser.add_argument("--check_name", type=str, default="last.pt", help="name of checkpoint")
args = parser.parse_args()

# data path
base_path = Path.cwd()
shs_data_path = os.path.join(base_path, "data", "shs-100k")
shs_pair_csv_path = os.path.join(shs_data_path, "pair_test", "_".join([args.csv_pref, "test.csv"]))
cover80_data_path = os.path.join(base_path, "data", "covers80_32k")
copyright40_data_path = os.path.join(base_path, "data", "copyright40", "CopyrightCases")
mcic_midi_path = os.path.join(base_path, "data", "MCIC", "MCIC DATA", "MIDI FILES")
datacos_data_path = os.path.join(base_path, "data", "da_tacos")

# csv path
copyright40_pair_csv_path = os.path.join(copyright40_data_path, "copyright40.csv")
mcic_pair_csv_path = os.path.join(base_path, "data", "MCIC", "MCIC DATA", "MCIC.csv")
datacos_csv_path = os.path.join(base_path, "data", "da_tacos", "da-tacos_metadata", "{}.csv".format("tri_test"))

cache_dir = os.path.join(base_path, args.cache_dir, "_".join([args.rep, args.cache_dir]))
cache_test_path = os.path.join(cache_dir, "shs100_test")
c80_cache_path = os.path.join(cache_dir, "covers80")
cp40_cache_path = os.path.join(cache_dir, "copyright40_Full")
mcic_cache_path = os.path.join(cache_dir, "MCIC")
datacos_cache_path = os.path.join(cache_dir, "da_tacos")

checkpoint_dir = Path("checkpoints") / args.exp_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_path = checkpoint_dir / "log_ci_eval.txt"

# load dataset
# shs_test_dataset = cs_shsdataset.SHSDataset(pair_csv_path=shs_pair_csv_path, dpath=shs_data_path, inp_rep=args.rep.upper(), crop_len=4375, split="test", cache_rep=True, cache_dir=cache_test_path)
# cover80_dataset = cs_cover80dataset.Cover80Dataset(lpath_src="list1.list", lpath_tgt="list2.list", dpath=cover80_data_path, inp_rep=args.rep.upper(), crop_len=4375, seed=2026, cache_rep=True, cache_dir=c80_cache_path)
copyright40_full_dataset = ci_copyright40dataset.Copyright40Dataset(pair_csv_path=copyright40_pair_csv_path, dpath=copyright40_data_path, inp_rep=args.rep.upper(), division="Full", crop_len=4375, cache_rep=True, cache_dir=cp40_cache_path)
mcic_dataset = ci_mcicdataset.MCICDataset(pair_csv_path=mcic_pair_csv_path, dpath=mcic_midi_path, inp_rep=args.rep.upper(), crop_len=4375, cache_rep=True, cache_dir=mcic_cache_path)
# datacos_dataset = pndatacosdataset.PNDATACOSDataset(csv_path=datacos_csv_path, dpath=datacos_data_path, inp_rep=args.rep.upper(), crop_len=4375, split="test", seed=2026, cache_dir=datacos_cache_path)

# shs_test_loader = DataLoader(shs_test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
# cover80_loader = DataLoader(cover80_dataset, batch_size=80, num_workers=4, shuffle=False, pin_memory=True)
copyright40_full_loader = DataLoader(copyright40_full_dataset, batch_size=40, shuffle=False, num_workers=4, pin_memory=True)
mcic_loader = DataLoader(mcic_dataset, batch_size=120, shuffle=False, num_workers=4, pin_memory=True)
# datacos_loader = DataLoader(datacos_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
tokenizer = a_vittokenizer.RawTokenizer(d_model=64*2, patch_h=23, patch_w=25)
selfattn = b_selfattention.SelfAttentionEncoder(nhead=8, d_model=64*2, num_layers=2, dim_feedforward=256*2)
model = c_self_sim_model.HPCPSelfAttentionModel(tokenizer=tokenizer, self_attention=selfattn, nhead=8, num_token=175, d_model=64*2).to(device=device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4,
)

best_loss = float("inf")

checkpoint_path = checkpoint_dir / args.check_name
# checkpoint_path = checkpoint_dir / "last.pt"
# checkpoint_path = checkpoint_dir / "best.pt"


if checkpoint_path.is_file():
    start_epoch, loaded_loss = load_checkpoint(
        model=model,
        optimizer=optimizer,
        path=checkpoint_path,
        device=device,
    )
    if loaded_loss is not None:
        best_loss = loaded_loss
else:
    print("No checkpoint found. Starting from epoch 1.")


# Evaluation on copy40 dataset pair
cp40_test_metrics = calculate_ci_metric(
    model=model,
    dataloader=copyright40_full_loader,
    device=device,
    save_path=checkpoint_dir,
    dset_name="cp40",
    epoch_name=args.check_name,
    # margin=0.2,
    # top_k=10
)

test_dataset = "copyright40"
log_ci_eval(log_path=log_path, dataset_name=test_dataset, metrics=cp40_test_metrics)

print(
    # f"Glob Pos Cos: {cp40_test_metrics['positive_cosine_mean']:.4f} | "
    # f"Glob Neg Cos: {cp40_test_metrics['negative_cosine_mean']:.4f} | "
    # f"Glob Gap: {cp40_test_metrics['cosine_gap']:.4f} | "
    f"Glob Roc_AUC {cp40_test_metrics['glob_auc']:.4f} | "
    f"Loc Roc_AUC {cp40_test_metrics['loc_auc']:.4f} | "
    f"Comb Roc_AUC {cp40_test_metrics['loc_auc']:.4f} | "
    f"Glob Threshold: {cp40_test_metrics['threshold']:.4f} | "
    f"Loc Threshold: {cp40_test_metrics['loc_threshold']:.4f} | "
    f"Comb Threshold: {cp40_test_metrics['loc_threshold']:.4f} | "
    f"Accuracy: {cp40_test_metrics['accuracy']:.4f} | "
    f"Loc Accuracy: {cp40_test_metrics['loc_accuracy']:.4f} | "
    f"Comb Accuracy: {cp40_test_metrics['loc_accuracy']:.4f} | "
    f"F1: {cp40_test_metrics['f1']:.4f} | "
    f"Loc F1: {cp40_test_metrics['loc_f1']:.4f} | "
    f"Comb F1: {cp40_test_metrics['loc_f1']:.4f} | "
    f"Precision: {cp40_test_metrics['precision']:.4f} | "
    f"Loc Precision: {cp40_test_metrics['loc_precision']:.4f} | "
    f"Comb Precision: {cp40_test_metrics['loc_precision']:.4f} | "
    f"Recall: {cp40_test_metrics['recall']:.4f} | "
    f"Loc Recall: {cp40_test_metrics['loc_recall']:.4f} | "
    f"Comb Recall: {cp40_test_metrics['loc_recall']:.4f}"
)

num_pos = sum(decision == 1 for decision in mcic_dataset.decisions)
num_neg = sum(decision == 0 for decision in mcic_dataset.decisions)

pos_weight = torch.tensor(
    [num_neg / num_pos],
    dtype=torch.float32,
    device=device,
)

# Evaluation on mcic dataset pair
mcic_test_metrics = calculate_ci_metric(
    model=model,
    dataloader=mcic_loader,
    device=device,
    save_path=checkpoint_dir,
    dset_name="mcic",
    epoch_name=args.check_name,
    # margin=0.2,
    # top_k=10
)

test_dataset = "mcic"
log_ci_eval(log_path=log_path, dataset_name=test_dataset, metrics=mcic_test_metrics)

print(
    # f"Glob Pos Cos: {mcic_test_metrics['positive_cosine_mean']:.4f} | "
    # f"Glob Neg Cos: {mcic_test_metrics['negative_cosine_mean']:.4f} | "
    # f"Glob Gap: {mcic_test_metrics['cosine_gap']:.4f} | "
    f"Glob Roc_AUC {mcic_test_metrics['glob_auc']:.4f} | "
    f"Loc Roc_AUC {mcic_test_metrics['loc_auc']:.4f} | "
    f"Comb Roc_AUC {mcic_test_metrics['loc_auc']:.4f} | "
    f"Glob Threshold: {mcic_test_metrics['threshold']:.4f} | "
    f"Loc Threshold: {mcic_test_metrics['loc_threshold']:.4f} | "
    f"Comb Threshold: {mcic_test_metrics['loc_threshold']:.4f} | "
    f"Accuracy: {mcic_test_metrics['accuracy']:.4f} | "
    f"Loc Accuracy: {mcic_test_metrics['loc_accuracy']:.4f} | "
    f"Comb Accuracy: {mcic_test_metrics['loc_accuracy']:.4f} | "
    f"F1: {mcic_test_metrics['f1']:.4f} | "
    f"Loc F1: {mcic_test_metrics['loc_f1']:.4f} | "
    f"Comb F1: {mcic_test_metrics['loc_f1']:.4f} | "
    f"Precision: {mcic_test_metrics['precision']:.4f} | "
    f"Loc Precision: {mcic_test_metrics['loc_precision']:.4f} | "
    f"Comb Precision: {mcic_test_metrics['loc_precision']:.4f} | "
    f"Recall: {mcic_test_metrics['recall']:.4f} | "
    f"Loc Recall: {mcic_test_metrics['loc_recall']:.4f} | "
    f"Comb Recall: {mcic_test_metrics['loc_recall']:.4f}"
)


# # Evaluation on datacos dataset pair
# datacos_test_metrics = evaluate_datacos(
#     model=model,
#     dataloader=datacos_loader,
#     device=device,
#     top_k=10
# )

# test_dataset = "datacos"
# log_eval(log_path=log_path, dataset_name=test_dataset, metrics=datacos_test_metrics)

# print(
#     f"Loss: {datacos_test_metrics['loss']:.4f} | "
#     f"Pos Cos: {datacos_test_metrics['positive_cosine_mean']:.4f} | "
#     f"Neg Cos: {datacos_test_metrics['negative_cosine_mean']:.4f} | "
#     f"Gap: {datacos_test_metrics['cosine_gap']:.4f} | "
#     f"Roc_AUC {datacos_test_metrics['roc_auc']:.4f} | "
#     f"mrr: {datacos_test_metrics['mrr']} | "
#     f"map: {datacos_test_metrics['map']} | "
#     f"top1_acc : {datacos_test_metrics['top1']} | "
#     f"topk_acc: {datacos_test_metrics['topk']}"
# )
