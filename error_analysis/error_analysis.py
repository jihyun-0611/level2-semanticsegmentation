import pandas as pd
import os
import numpy as np
import json
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import *
from config.config import Config
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/data/ephemeral/home/lv2-ss/experiments/completed/23_DUCKNet.yaml')
    parser.add_argument('--add_wandb', type=bool, default=True)
    args, _ = parser.parse_known_args()
    
    if args.add_wandb:
        parser.add_argument('--id', type=str, default='12w5vgm3')
    
    
    return parser.parse_args()

class DataProcessor:
    def __init__(self, config):
        """
        데이터 처리를 위한 클래스 초기화
        """
        self.config = config
        self.classes = config.DATA.CLASSES
        self.num_classes = len(self.classes)
        self.image_shape = (2048, 2048)
        self.CLASS2IND = {v: i for i, v in enumerate(self.classes)}
        self.IND2CLASS = {i: v for i, v in enumerate(self.classes)}

    def create_gt_mask(self, label_path):
        """GT 마스크 생성 함수"""
        label = np.zeros((self.num_classes, self.image_shape[0], self.image_shape[1]), 
                        dtype=np.uint8)
        
        with open(label_path, "r") as f:
            annotations = json.load(f)["annotations"]
        
        class_points = {i: [] for i in range(self.num_classes)}
        for ann in annotations:
            class_ind = self.CLASS2IND[ann["label"]]
            class_points[class_ind].append(np.array(ann["points"]))
        
        for class_ind, points_list in class_points.items():
            if points_list:
                class_mask = np.zeros(self.image_shape[:2], dtype=np.uint8)
                cv2.fillPoly(class_mask, points_list, 1)
                label[class_ind] = class_mask
        
        return label

    def create_pred_mask(self, pred_df, image_name):
        """예측 마스크 생성 함수"""
        pred_masks = np.zeros((self.num_classes, self.image_shape[0], self.image_shape[1]))
        
        image_preds = pred_df[pred_df['image_name'] == image_name]
        valid_preds = image_preds[pd.notna(image_preds['rle'])]
        
        if not valid_preds.empty:
            for _, row in valid_preds.iterrows():
                class_idx = self.CLASS2IND[row['class']]
                mask = decode_rle_to_mask(row['rle'], self.image_shape[0], self.image_shape[1])
                pred_masks[class_idx] = mask
                
        return pred_masks

    def analyze_single_image(self, gt_path, pred_path, pred_df):
        """단일 이미지에 대한 종합적 분석"""
        label_mask = self.create_gt_mask(gt_path)
        pred_mask = self.create_pred_mask(pred_df, pred_path)
        
        analysis = {
            'image_id': pred_path,
            'confusion_matrix': self.calc_confusion_matrix(label_mask, pred_mask),
            'class_metrics': self.calculate_class_metrics(label_mask, pred_mask),
            'error_patterns': self.analyze_error_patterns(label_mask, pred_mask),
            'boundary_analysis': self.analyze_boundary_errors(label_mask, pred_mask)
        }
        return analysis

    def calc_confusion_matrix(self, label_mask, pred_mask):
        """혼동 행렬 계산"""
        num_classes = label_mask.shape[0]
        label_flat = label_mask.reshape(num_classes, -1)
        pred_flat = pred_mask.reshape(num_classes, -1)
        return label_flat @ pred_flat.T

    def calculate_class_metrics(self, label_mask, pred_mask):
        """클래스별 성능 지표 계산"""
        metrics = {}
        
        # numpy를 torch tensor로 변환 (B, C, H, W) 형태로
        label = torch.from_numpy(label_mask).float().unsqueeze(0)  # (1, C, H, W)
        pred = torch.from_numpy(pred_mask).float().unsqueeze(0)    # (1, C, H, W)
        
        # 전체 클래스에 대한 dice 점수 계산
        dices = dice_coef(pred, label)  # (C,) 형태의 텐서 반환
        dices = torch.mean(dices, dim=0)  # 배치 차원에 대해 평균 계산
        
        for class_idx in range(self.num_classes):
            # 각 클래스별 2D 마스크
            label_2d = label_mask[class_idx]
            pred_2d = pred_mask[class_idx]
            
            # Precision & Recall 계산
            tp = np.sum((pred_2d > 0.5) & (label_2d > 0.5))
            fp = np.sum((pred_2d > 0.5) & (label_2d < 0.5))
            fn = np.sum((pred_2d < 0.5) & (label_2d > 0.5))
            
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            
            metrics[self.IND2CLASS[class_idx]] = {
                'dice': float(dices[class_idx].numpy()),  # numpy()로 변환 후 float로 변환
                'precision': precision,
                'recall': recall,
                'size': np.sum(label_2d > 0.5)
            }
        
        return metrics
    

    def analyze_error_patterns(self, label_mask, pred_mask):
        """오류 패턴 분석"""
        error_patterns = {}
        
        for class_idx in range(self.num_classes):
            label = label_mask[class_idx]
            pred = pred_mask[class_idx]
            
            fp = np.sum((pred > 0.5) & (label == 0))
            fn = np.sum((pred < 0.5) & (label == 1))
            
            label_size = np.sum(label)
            pred_size = np.sum(pred > 0.5)
            size_error = (pred_size - label_size) / (label_size + 1e-7)
            
            label_components = cv2.connectedComponents(label.astype(np.uint8))[0]
            pred_components = cv2.connectedComponents((pred > 0.5).astype(np.uint8))[0]
            
            error_patterns[self.IND2CLASS[class_idx]] = {
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'size_error': float(size_error),
                'fragmentation': pred_components - label_components
            }
        return error_patterns

    def analyze_boundary_errors(self, label_mask, pred_mask):
        """경계 오류 분석"""
        boundary_errors = {}
        kernel = np.ones((3,3), np.uint8)
        
        for class_idx in range(self.num_classes):
            label = label_mask[class_idx]
            pred = pred_mask[class_idx] > 0.5
            
            label_boundary = cv2.dilate(label, kernel) - cv2.erode(label, kernel)
            boundary_error = np.sum((pred != label) & (label_boundary > 0))
            total_boundary = np.sum(label_boundary)
            
            boundary_errors[self.IND2CLASS[class_idx]] = {
                'boundary_error_rate': float(boundary_error / (total_boundary + 1e-7)),
                'total_boundary_pixels': int(total_boundary)
            }
        return boundary_errors

def align_names(gt_paths, image_names):
    """경로 정렬 함수"""
    gt_image_ids = [path.split("/")[1][:-5] for path in gt_paths]
    image_ids = [name.split(".")[0] for name in image_names]
    
    gt_dict = {img_id: path for img_id, path in zip(gt_image_ids, gt_paths)}
    image_dict = {img_id: name for img_id, name in zip(image_ids, image_names)}
    
    common_ids = sorted(set(gt_image_ids) & set(image_ids))
    
    sorted_gt_paths = [gt_dict[img_id] for img_id in common_ids]
    sorted_image_names = [image_dict[img_id] for img_id in common_ids]
    
    return sorted_gt_paths, sorted_image_names

def visualize_results(processor, analysis_results, save_dir='./', args=None):
    """결과 시각화 및 wandb 로깅"""
    if args.add_wandb:
        wandb.init(
            project=processor.config.WANDB.PROJECT_NAME,
            entity=processor.config.WANDB.ENTITY, 
            id=args.id, 
            resume="must")
    
    # 1. Confusion Matrix
    total_cm = np.zeros((processor.num_classes, processor.num_classes))
    for result in analysis_results:
        total_cm += result['confusion_matrix']
    
    plt.figure(figsize=(20, 16))
    norm_cm = total_cm / total_cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(norm_cm,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                square=True,
                xticklabels=processor.classes,
                yticklabels=processor.classes)
    
    plt.title('Multi-label Segmentation Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('GroundTruth')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 저장 및 wandb 로깅
    if save_dir:
        modelname = os.path.splitext(processor.config.MODEL.MODEL_NAME)[0]
        save_path = os.path.join(save_dir, f'{modelname}_confusion_matrix.png')
        plt.savefig(save_path)
        if args.add_wandb:
            wandb.log({"error_analysis/confusion_matrix": wandb.Image(save_path)})
    plt.close()

    # 2. Class-wise Performance
    class_metrics = pd.DataFrame([
        {
            'class': class_name,
            **metrics
        }
        for result in analysis_results
        for class_name, metrics in result['class_metrics'].items()
    ])
    
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=class_metrics, x='class', y='dice', color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Dice Scores by Bone Class')
    
    # 저장 및 wandb 로깅
    if save_dir:
        save_path = os.path.join(save_dir, f'{modelname}_dice_distribution.png')
        plt.savefig(save_path)
        if args.add_wandb:
            wandb.log({"error_analysis/dice_distribution": wandb.Image(save_path)})
    plt.close()

    # 3. Error Pattern Analysis
    error_summary = pd.DataFrame([
        {
            'class': class_name,
            'boundary_error_rate': errors['boundary_error_rate']
        }
        for result in analysis_results
        for class_name, errors in result['boundary_analysis'].items()
    ])
    
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=error_summary, x='class', y='boundary_error_rate', color='lightgreen')
    plt.xticks(rotation=45, ha='right')
    plt.title('Boundary Error Rates by Bone Class')
    
    # 저장 및 wandb 로깅
    if save_dir:
        save_path = os.path.join(save_dir, f'{modelname}_boundary_errors.png')
        plt.savefig(save_path)
        if args.add_wandb:
            wandb.log({"error_analysis/boundary_errors": wandb.Image(save_path)})
    plt.close()

    # wandb 종료
    if args.add_wandb:
        wandb.finish()


def main(config, args):
    """메인 실행 함수"""
    processor = DataProcessor(config)
    
    # 데이터 로드
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
    
    # 병렬 처리로 분석 실행
    with ThreadPoolExecutor(max_workers=8) as executor:
        analysis_results = list(tqdm(
            executor.map(
                lambda args: processor.analyze_single_image(*args),
                [(os.path.join(json_root, gt), img, pred_df) 
                 for gt, img in zip(sorted_gt_paths, sorted_image_names)]
            ),
            total=len(sorted_gt_paths),
            desc="Analyzing images"
        ))
    
    # 결과 시각화
    save_dir = os.path.join(config.TRAIN.OUTPUT_DIR, 'error_analysis')
    os.makedirs(save_dir, exist_ok=True)
    visualize_results(processor, analysis_results, save_dir, args)

if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config)
    main(config, args)