#!/usr/bin/bash

for epoch in $(seq 5 5 50); do
    printf -v ep "%03d" "$epoch"
    python 3_eval_hpcp_attn.py --check_name="model_epoch_${ep}.pt"
done
