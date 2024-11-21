#!/bin/bash

# experiments 디렉토리의 모든 yaml 파일을 순회
for config in experiments/completed/{11,12,13,14,15,16,17,18}*.yaml
do
    echo "Running experiment with config: $config"
    python evaluation.py --config "$config"
    echo "----------------------------------------"
done