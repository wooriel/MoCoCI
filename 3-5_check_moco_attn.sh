#!/usr/bin/bash

for epoch in $(seq 5 5 35); do
    printf -v ep "%03d" "$epoch"
    python 3-5_eval_moco_attn.py --check_name="model_epoch_${ep}.pt"
done
