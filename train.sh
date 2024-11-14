#!/bin/bash

# 단일 실험을 위한 스크립트
CONFIG=${1:-"experiments/test.yaml"} 

echo "Running single experiment with config: $CONFIG"
python train.py --config "$CONFIG"
echo "Experiment completed!"