#!/usr/bin/env python3

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import gc
import warnings
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

from sklearn.metrics import confusion_matrix

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rich.console import Console
from rich.table import Table
from rich.progress import track

# ------------------ Setup ------------------
console = Console()
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------ CLAHE ------------------
class CLAHETransform(ImageOnlyTransform):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image, **params):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def get_val_transform():
    return A.Compose([
        CLAHETransform(),
        A.Resize(640, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# ------------------ Dataset ------------------
class RSODDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        from glob import glob
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.img_files = sorted(glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.png')))
        self.class_dict = {"aircraft": 1, "oiltank": 2, "overpass": 3, "playground": 4}
        self.id_to_name = {v: k for k, v in self.class_dict.items()}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        xml = os.path.join(self.ann_dir, os.path.basename(img_path).replace(".jpg", ".xml").replace(".png", ".xml"))
        root = ET.parse(xml).getroot()

        boxes, labels = [], []
        for obj in root.findall('object'):
            cls = obj.find('name').text.lower()
            if cls not in self.class_dict:
                continue
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[cls])

        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4))
        labels = np.array(labels) if labels else np.zeros(0, )

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        return image, target

# ------------------ Soft-NMS ------------------
def soft_nms_pytorch(boxes, scores, iou_threshold=0.5):
    from torchvision.ops import nms
    return nms(boxes, scores, iou_threshold)

# ------------------ Model ------------------
def get_model(num_classes):
    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    model = retinanet_resnet50_fpn_v2(weights=weights)
    in_feat = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(in_feat, num_anchors, num_classes)
    return model

# ------------------ Utilities ------------------
def box_iou(boxes1, boxes2):
    area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
    area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

# ------------------ Main ------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.rule("[bold cyan]RetinaNet Inference & mAP Evaluation")

    dataset = RSODDataset(args.dataset_dir, args.ann_dir, transforms=get_val_transform())
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_classes=5)  # 4 classes + background
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    all_true_labels = []
    all_pred_labels = []

    class_names = ["background", "aircraft", "oiltank", "overpass", "playground"]

    results = []

    with torch.no_grad():
        for idx, (imgs, targets) in enumerate(track(val_loader, description="🔍 Running inference...")):
            img = imgs[0].to(device)
            target = targets[0]

            output = model([img])[0]

            pred_boxes = output['boxes'].cpu()
            pred_scores = output['scores'].cpu()
            pred_labels = output['labels'].cpu()

            keep = soft_nms_pytorch(pred_boxes, pred_scores)
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]

            true_boxes = target['boxes']
            true_labels = target['labels']

            # Save predictions for COCO eval format
            img_id = idx
            for b, s, l in zip(pred_boxes, pred_scores, pred_labels):
                x1, y1, x2, y2 = b
                results.append({
                    "image_id": img_id,
                    "category_id": int(l),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s)
                })

            # Match preds with GT to compute confusion matrix
            ious = box_iou(true_boxes, pred_boxes) if (len(true_boxes)>0 and len(pred_boxes)>0) else torch.empty((len(true_boxes), len(pred_boxes)))

            assigned_pred = torch.zeros(len(true_boxes), dtype=torch.long)  # predicted labels per GT
            assigned_pred.fill_(0)  # 0 = background / no match
            matched_pred_idx = set()

            for i in range(len(true_boxes)):
                if len(pred_boxes) == 0:
                    break
                iou_row = ious[i]
                max_iou, max_j = torch.max(iou_row, dim=0)
                if max_iou >= 0.5 and max_j.item() not in matched_pred_idx:
                    assigned_pred[i] = pred_labels[max_j]
                    matched_pred_idx.add(max_j.item())
                else:
                    assigned_pred[i] = 0

            all_true_labels.extend(true_labels.tolist())
            all_pred_labels.extend(assigned_pred.tolist())

            unmatched_preds = [j for j in range(len(pred_boxes)) if j not in matched_pred_idx]
            all_true_labels.extend([0]*len(unmatched_preds))
            all_pred_labels.extend(pred_labels[unmatched_preds].tolist())

    # Save predictions JSON for COCO evaluation
    with open("retina_predictions.json", "w") as f:
        json.dump(results, f)

    # Prepare COCO GT format
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {}, "licenses": [],
        "images": [{"id": i} for i in range(len(dataset))],
        "categories": [{"id": 1, "name": "aircraft"},
                       {"id": 2, "name": "oiltank"},
                       {"id": 3, "name": "overpass"},
                       {"id": 4, "name": "playground"}],
        "annotations": []
    }

    ann_id = 0
    for i in range(len(dataset)):
        _, target = dataset[i]
        for b, l in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = b.numpy()
            coco_gt.dataset['annotations'].append({
                "id": ann_id,
                "image_id": i,
                "category_id": int(l),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0
            })
            ann_id += 1

    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes("retina_predictions.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    # Overall evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    overall_map = coco_eval.stats[1]  # mAP@0.5

    # Per-class mAP@0.5
    class_maps = {}
    for cat_id in range(1, 5):
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        class_maps[dataset.id_to_name[cat_id]] = coco_eval.stats[1]

    # Show mAP table
    table = Table(title="📊 Mean Average Precision (mAP@0.5)")
    table.add_column("Class")
    table.add_column("mAP@0.5", justify="center")
    for cls, ap in class_maps.items():
        table.add_row(cls, f"{ap:.4f}")
    table.add_row("[bold]Overall[/bold]", f"[bold]{overall_map:.4f}[/bold]")
    console.print(table)

    # Confusion matrix display
    cm = confusion_matrix(all_true_labels, all_pred_labels, labels=[0,1,2,3,4])
    console.print("\nConfusion Matrix:")
    console.print(cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--ann_dir", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    main(args)
