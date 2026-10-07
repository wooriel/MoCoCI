#!/usr/bin/bash

for epoch in $(seq 5 5 75); do
    printf -v ep "%03d" "$epoch"
    python 4-5_eval_ci_moco_attn.py --check_name="model_epoch_${ep}.pt"
done
