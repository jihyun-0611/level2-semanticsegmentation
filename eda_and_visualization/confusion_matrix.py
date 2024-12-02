import pandas as pd
import glob
import os
from PIL import Image, ImageDraw
import numpy as np
import json
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from utils.metrics import *

from config.config import Config
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./experiments/completed/23_DUCKNet.yaml')
    return parser.parse_args()

class DataProcessor: 
    def __init__(self, config):
        self.config = config
        self.classes = config.DATA.CLASSES
        self.num_classes = len(self.classes)
        self.image_shape = (2048, 2048)
        self.CLASS2IND = {v: i for i, v in enumerate(self.classes)}
        self.IND2CLASS = {i: v for i, v in enumerate(self.classes)}

    # GT 마스크 생성 함수
    def create_gt_mask(self, label_path):
        # GT 마스크의 shape 생성 (높이, 너비, 클래스 수)
        label = np.zeros((self.num_classes, self.image_shape[0], self.image_shape[1]), dtype=np.uint8)
        
        # JSON 레이블 파일 열기
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
        
        class_points = {i: [] for i in range(self.num_classes)}
        # 클래스별로 마스크 생성
        for ann in annotations:
            class_ind = self.CLASS2IND[ann["label"]]
            class_points[class_ind].append(np.array(ann["points"]))
            
        # 각 클래스별로 한 번에 모든 폴리곤 처리
        for class_ind, points_list in class_points.items():
            if points_list:
                class_mask = np.zeros(self.image_shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_mask, points_list, 1)
                label[class_ind] = class_mask
    
        return label

    def create_pred_mask(self, pred_df, image_name):
        # rle 예측값을 마스크로 변환        
        pred_masks = np.zeros((self.num_classes, self.image_shape[0], self.image_shape[1]))
        
        # 이미지별 예측값 필터링
        image_preds = pred_df[pred_df['image_name'] == image_name]
        valid_preds = image_preds[pd.notna(image_preds['rle'])]
        
        if not valid_preds.empty:
            for _, row in valid_preds.iterrows():
                class_idx = self.CLASS2IND[row['class']]
                mask = decode_rle_to_mask(row['rle'], self.image_shape[0], self.image_shape[1])
                pred_masks[class_idx] = mask
                
        return pred_masks

def align_names(gt_paths, image_names):
    # gt_paths에서 이미지 ID 추출
    gt_image_ids = [path.split("/")[1][:-5] for path in gt_paths]  # image1666141916583
    # image_names에서 이미지 ID 추출
    image_ids = [name.split(".")[0] for name in image_names]  # image1666141916583
    
    # 매칭을 위한 딕셔너리 생성
    gt_dict = {img_id: path for img_id, path in zip(gt_image_ids, gt_paths)}
    image_dict = {img_id: name for img_id, name in zip(image_ids, image_names)}
    
    # 공통된 이미지 ID 찾기
    common_ids = sorted(set(gt_image_ids) & set(image_ids))
    
    # 공통 ID를 기준으로 정렬된 리스트 생성
    sorted_gt_paths = [gt_dict[img_id] for img_id in common_ids]
    sorted_image_names = [image_dict[img_id] for img_id in common_ids]
    
    return sorted_gt_paths, sorted_image_names


def calc_confusion_matrix(label_mask, pred_mask):
    num_classes = label_mask.shape[0]
    
    # 마스크를 1차원으로 재구성
    label_flat = label_mask.reshape(num_classes, -1)
    pred_flat = pred_mask.reshape(num_classes, -1)
    
    # 행렬 곱셈으로 혼동 행렬 계산
    confusion_matrix = label_flat @ pred_flat.T
    
    return confusion_matrix

def process_single_image(args):
    """
    단일 이미지 처리 함수 (병렬 처리용)
    """
    gt_path, pred_path, pred_df, processor = args
    label_mask = processor.create_gt_mask(gt_path)
    pred_mask = processor.create_pred_mask(pred_df, pred_path)
    return calc_confusion_matrix(label_mask, pred_mask)

def calc_all_data(label_paths, pred_paths, pred_df, processor,
                  image_shape = (2048,2048), 
                  json_root = '/data/ephemeral/home/data/train/outputs_json',
                  max_workers = 8):
    all_cm = np.zeros((processor.num_classes, processor.num_classes))
    
    process_args = [(os.path.join(json_root, label_path), 
                    pred_path, 
                    pred_df, 
                    processor) 
                   for label_path, pred_path in zip(label_paths, pred_paths)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_image, process_args),
            total=len(process_args),
            desc="Computing confusion matrix"
        ))
    
    # 결과 합산
    all_cm = np.sum(results, axis=0)
    
    return all_cm

def plot_confusion_matrix(confusion_matrix, class_names, normalize=True, save_path='./'):
    """
    Confusion matrix를 heatmap으로 시각화
    
    Args:
        confusion_matrix: (num_classes, num_classes) 형태의 confusion matrix
        class_names: 클래스 이름 리스트 (옵션)
        normalize: 행 기준 정규화 여부
        save_path: 저장할 경로 (옵션)
    """
    if normalize:
        # 행 기준으로 정규화 (각 true class에 대한 예측 분포)
        norm_cm = np.zeros_like(confusion_matrix, dtype=float)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        norm_cm = np.divide(confusion_matrix, row_sums,
                          where=row_sums!=0) * 100
    else: norm_cm = confusion_matrix

    plt.figure(figsize=(20, 16))
    
    # Heatmap 생성
    sns.heatmap(norm_cm, 
                annot=True, 
                fmt='.1f' if normalize else '.0f',
                cmap='Blues',
                square=True,
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Multi-label Segmentation Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('GroundTruth')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()    
    
    if save_path:
        modelname = os.path.splitext(config.MODEL.MODEL_NAME)[0]
        plt.savefig(os.path.join(save_path, modelname))
    plt.show()

        
    
def main(config):
    processor = DataProcessor(config)
    
    # CSV 파일 로드
    csv_path = os.path.join(config.TRAIN.OUTPUT_DIR, config.TRAIN.CSV_NAME)
    pred_df = pd.read_csv(csv_path)
    
    # GT 경로 생성
    json_root = config.DATA.LABEL_ROOT
    image_names = set(pred_df['image_name'])
    json_names = [os.path.splitext(name)[0] + '.json' for name in image_names]
    
    ID2GT = {}
    for folder_name in os.listdir(json_root):
        folder_path = os.path.join(json_root, folder_name)
        if os.path.isdir(folder_path):
            for label_name in os.listdir(folder_path):
                if label_name in json_names:
                    if folder_name not in ID2GT:
                        ID2GT[folder_name] = []
                    ID2GT[folder_name].append(label_name)
    
    gt_paths = []
    for id in ID2GT.keys():
        gt_paths.extend([os.path.join(id, path) for path in ID2GT[id]])
    
    # 경로 정렬
    sorted_gt_paths, sorted_image_names = align_names(gt_paths, list(image_names))
    
    # 혼동 행렬 계산
    cm = calc_all_data(
        label_paths=sorted_gt_paths,
        pred_paths=sorted_image_names,
        pred_df=pred_df,
        processor=processor
    )
    plot_confusion_matrix(cm, class_names=config.DATA.CLASSES, normalize=True)

if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config)
    main(config)