#!/bin/bash

# 단일 실험을 위한 스크립트
CONFIG=${1:-"experiments/test.yaml"} 

echo "Running inference with config: $CONFIG"
python inference.py --config "$CONFIG"
echo "Inference completed!"