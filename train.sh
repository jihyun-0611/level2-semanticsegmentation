#!/bin/bash

# experiments 디렉토리의 모든 yaml 파일을 순회
for config in experiments/*.yaml
do
    echo "Running experiment with config: $config"
    python train.py --config "$config"
    echo "----------------------------------------"
done