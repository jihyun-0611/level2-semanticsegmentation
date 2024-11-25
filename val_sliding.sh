#!/bin/bash

# 단일 실험을 위한 스크립트
CONFIG=${1:-"experiments/Finetuning_sliding_window.yaml"} 

echo "Running single experiment with config: $CONFIG"
python val_sliding.py --config "$CONFIG"
echo "Experiment completed!"