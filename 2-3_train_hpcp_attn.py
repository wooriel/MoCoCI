import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import pndatacosdataset
from models import a_vittokenizer, b_selfattention, c_self_sim_model
from train_align import train_vit_align_one_epoch, validate_vit_align_one_epoch, log_train_val
from checkpoint import save_checkpoint, load_checkpoint
from data_augmentation import cyclic_subset


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="da_tacos", help="data directory name")
parser.add_argument("--csv_pref", type=str, default="pair")
parser.add_argument("--exp_name", type=str, default="hpcp_attn_v13", help="name of training experiment")
parser.add_argument("--csv_dir", type=str, default="da-tacos_metadata", help="directory of metadata")
# parser.add_argument("--split", type=str, default="train")
parser.add_argument("--cache_dir", type=str, default="hpcp_cache", help="cache directory name")
parser.add_argument("--rep", type=str, default="hpcp", help="representation of the input")
args = parser.parse_args()

# path
base_path = Path.cwd()
data_path = os.path.join(base_path, "data", args.data_dir)
csv_dir = os.path.join(base_path, "data", args.data_dir, args.csv_dir)
csv_train_path = os.path.join(csv_dir, "{}.csv".format("tri_train"))
csv_valid_path = os.path.join(csv_dir, "{}.csv".format("tri_validate"))
cache_path = os.path.join(base_path, "cache", args.cache_dir, args.data_dir) # cache / hpcp_cache

checkpoint_dir = Path("checkpoints") / args.exp_name
checkpoint_dir.mkdir(parents=True, exist_ok=True)
log_path = checkpoint_dir / "log_train.txt"

# load dataset
datacos_train_dataset = pndatacosdataset.PNDATACOSDataset(csv_path=csv_train_path, dpath=data_path, inp_rep=args.rep.upper(), crop_len=4375, split="train", seed=2026, cache_dir=cache_path)
datacos_val_dataset = pndatacosdataset.PNDATACOSDataset(csv_path=csv_valid_path, dpath=data_path, inp_rep=args.rep.upper(), crop_len=4375, split="validate", seed=2026, cache_dir=cache_path)

datacos_train_loader = DataLoader(datacos_train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
datacos_val_loader = DataLoader(datacos_val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True) # pin_memory batches into page-locked cpu ram (fast transfer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
tokenizer = a_vittokenizer.RawTokenizer(d_model=64*2, patch_h=23, patch_w=25)
selfattn = b_selfattention.SelfAttentionEncoder(nhead=8, d_model=64*2, num_layers=2, dim_feedforward=256*2, dropout=0.2)
model = c_self_sim_model.HPCPSelfAttentionModel(tokenizer=tokenizer, self_attention=selfattn, nhead=8, num_token=175, d_model=64*2).to(device=device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4,
)

# training
num_epochs = 300
validate_every = 1
checkpoint_every = 5

start_epoch = 1
best_val_auc = -float("inf")

# checkpoint_path = checkpoint_dir / "best.pt"
checkpoint_path = checkpoint_dir / "last.pt"

if checkpoint_path.is_file():
    start_epoch, loaded_val_loss = load_checkpoint(
        model=model,
        optimizer=optimizer,
        path=checkpoint_path,
        device=device,
    )
    if loaded_val_loss is not None:
        best_val_auc = loaded_val_loss
else:
    print("No checkpoint found. Starting from epoch 1.")


for epoch in range(start_epoch, num_epochs + 1):
    # cyclic_train_dataset = cyclic_subset(datacos_train_dataset, epoch=epoch, ratio=0.05)
    # cyclic_train_loader = DataLoader(cyclic_train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    train_metrics = train_vit_align_one_epoch(
        model=model,
        dataloader=datacos_train_loader,
        optimizer=optimizer,
        device=device,
        glo_margin=0.2,
        loc_margin=0.2,
    )

    print(
        f"Epoch [{epoch}/{num_epochs}] | "
        f"Loss: {train_metrics['loss']:.4f} | "
        f"Pos Cos: {train_metrics['positive_cosine_mean']:.4f} | "
        f"Neg Cos: {train_metrics['negative_cosine_mean']:.4f} | "
        f"Gap: {train_metrics['cosine_gap']:.4f} | "
        f"tri_loss: {train_metrics['tri_loss']:.4f} | ",
        f"ap_loss: {train_metrics['ap_loss']:.4f} | ",
        f"local_loss: {train_metrics['local_loss']:.4f}"
    )

    # Validation
    if epoch % validate_every == 0:
        # cyclic_val_dataset = cyclic_subset(datacos_val_dataset, epoch=epoch, ratio=0.05)
        # cyclic_val_loader = DataLoader(cyclic_val_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        val_metrics = validate_vit_align_one_epoch(
            model=model,
            dataloader=datacos_val_loader,
            device=device,
            glo_margin=0.2,
            loc_margin=0.2,
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=checkpoint_dir / "best.pt",
                val_loss=best_val_auc,
            )

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Pos: {val_metrics['positive_cosine_mean']:.4f} | "
            f"Neg: {val_metrics['negative_cosine_mean']:.4f} | "
            f"Gap: {val_metrics['cosine_gap']:.4f} | "
            f"AUC {val_metrics['roc_auc']:.4f} | "
            f"MRR: {val_metrics['mrr']:.4f} | ",
            f"MAP: {val_metrics['map']:.4f} | ",
            f"Top1: {val_metrics['top1']:.4f} | ",
            f"Topk: {val_metrics['topk']:.4f} | ",
            f"tri_loss: {val_metrics['tri_loss']:.4f} | ",
            f"ap_loss: {val_metrics['ap_loss']:.4f} | ",
            f"local_loss: {val_metrics['local_loss']:.4f}"
        )

        log_train_val(log_path=log_path, epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics)

    # -------------------------
    # Checkpoint
    # -------------------------
    if epoch % checkpoint_every == 0:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            path=checkpoint_dir / "model_epoch_{:03d}.pt".format(epoch),
        )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        path=checkpoint_dir / "last.pt",
    )
