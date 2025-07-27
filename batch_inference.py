# #!/usr/bin/env python3

# import os
# import json
# import argparse
# import torch
# from torch.utils.data import DataLoader, Dataset
# from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
# from torchvision.models.detection.retinanet import RetinaNetClassificationHead

# from PIL import Image
# import numpy as np
# import xml.etree.ElementTree as ET
# import gc
# import warnings
# import cv2

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from albumentations.core.transforms_interface import ImageOnlyTransform

# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

# from rich.console import Console
# from rich.table import Table
# from rich.progress import track

# import io
# import contextlib

# # ------------------ Setup ------------------
# console = Console()
# warnings.filterwarnings("ignore", category=UserWarning)

# # ------------------ CLAHE ------------------
# class CLAHETransform(ImageOnlyTransform):
#     def __init__(self, clip_limit=2.0, tile_grid_size=(8,8), always_apply=False, p=1.0):
#         super().__init__(always_apply, p)
#         self.clip_limit = clip_limit
#         self.tile_grid_size = tile_grid_size

#     def apply(self, image, **params):
#         lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
#         l, a, b = cv2.split(lab)
#         clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
#         cl = clahe.apply(l)
#         merged = cv2.merge((cl, a, b))
#         return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

# def get_val_transform():
#     return A.Compose([
#         CLAHETransform(),
#         A.Resize(640, 640),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#     ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# # ------------------ Dataset ------------------
# class RSODDataset(Dataset):
#     def __init__(self, img_dir, ann_dir, transforms=None):
#         from glob import glob
#         self.img_dir = img_dir
#         self.ann_dir = ann_dir
#         self.transforms = transforms
#         self.img_files = sorted(glob(os.path.join(img_dir, '*.jpg')) + glob(os.path.join(img_dir, '*.png')))
#         self.class_dict = {"aircraft": 1, "oiltank": 2, "overpass": 3, "playground": 4}
#         self.id_to_name = {v: k for k, v in self.class_dict.items()}

#     def __len__(self):
#         return len(self.img_files)

#     def __getitem__(self, idx):
#         img_path = self.img_files[idx]
#         image = np.array(Image.open(img_path).convert("RGB"))
#         xml = os.path.join(self.ann_dir, os.path.basename(img_path).replace(".jpg", ".xml").replace(".png", ".xml"))
#         root = ET.parse(xml).getroot()

#         boxes, labels = [], []
#         for obj in root.findall('object'):
#             cls = obj.find('name').text.lower()
#             if cls not in self.class_dict:
#                 continue
#             bbox = obj.find('bndbox')
#             xmin = float(bbox.find('xmin').text)
#             ymin = float(bbox.find('ymin').text)
#             xmax = float(bbox.find('xmax').text)
#             ymax = float(bbox.find('ymax').text)
#             boxes.append([xmin, ymin, xmax, ymax])
#             labels.append(self.class_dict[cls])

#         boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4))
#         labels = np.array(labels) if labels else np.zeros(0, )

#         if self.transforms:
#             transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
#             image = transformed['image']
#             boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
#             labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
#         else:
#             image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
#             boxes = torch.tensor(boxes, dtype=torch.float32)
#             labels = torch.tensor(labels, dtype=torch.int64)

#         target = {'boxes': boxes, 'labels': labels}
#         return image, target

# # ------------------ Soft-NMS placeholder ------------------
# from torchvision.ops import nms
# def soft_nms_pytorch(boxes, scores, iou_threshold=0.5):
#     return nms(boxes, scores, iou_threshold)

# # ------------------ Model ------------------
# def get_model(num_classes):
#     weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
#     model = retinanet_resnet50_fpn_v2(weights=weights)
#     in_feat = model.head.classification_head.conv[0][0].in_channels
#     num_anchors = model.head.classification_head.num_anchors
#     model.head.classification_head = RetinaNetClassificationHead(in_feat, num_anchors, num_classes)
#     return model

# # ------------------ Main ------------------
# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     console.rule("[bold cyan]RetinaNet Inference")

#     dataset = RSODDataset(args.dataset_dir, args.ann_dir, transforms=get_val_transform())
#     val_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

#     model = get_model(num_classes=5)
#     model.load_state_dict(torch.load(args.weights, map_location=device))
#     model.to(device)
#     model.eval()

#     gc.collect()
#     torch.cuda.empty_cache()

#     results = []
#     with torch.no_grad():
#         for idx, (imgs, targets) in enumerate(track(val_loader, description="🔍 Running inference...")):
#             imgs = [img.to(device) for img in imgs]
#             outputs = model(imgs)
#             for j, output in enumerate(outputs):
#                 boxes = output['boxes'].cpu()
#                 scores = output['scores'].cpu()
#                 labels = output['labels'].cpu()
#                 keep = soft_nms_pytorch(boxes, scores)
#                 boxes = boxes[keep]
#                 scores = scores[keep]
#                 labels = labels[keep]
#                 for b, s, l in zip(boxes, scores, labels):
#                     x1, y1, x2, y2 = b
#                     results.append({
#                         "image_id": idx * len(outputs) + j,
#                         "category_id": int(l),
#                         "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
#                         "score": float(s)
#                     })

#     with open("retina_predictions.json", "w") as f:
#         json.dump(results, f)

#     # COCO GT build
#     coco_gt = COCO()
#     coco_gt.dataset = {
#         "info": {"year": 2025},
#         "licenses": [],
#         "images": [{"id": i} for i in range(len(dataset))],
#         "categories": [{"id": 1, "name": "aircraft"},
#                        {"id": 2, "name": "oiltank"},
#                        {"id": 3, "name": "overpass"},
#                        {"id": 4, "name": "playground"}],
#         "annotations": []
#     }
#     ann_id = 0
#     for i in range(len(dataset)):
#         _, target = dataset[i]
#         for b, l in zip(target['boxes'], target['labels']):
#             x1, y1, x2, y2 = b.numpy()
#             coco_gt.dataset['annotations'].append({
#                 "id": ann_id,
#                 "image_id": i,
#                 "category_id": int(l),
#                 "bbox": [x1, y1, x2 - x1, y2 - y1],
#                 "area": float((x2 - x1) * (y2 - y1)),
#                 "iscrowd": 0
#             })
#             ann_id += 1
#     coco_gt.createIndex()
#     coco_dt = coco_gt.loadRes("retina_predictions.json")
#     coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     with contextlib.redirect_stdout(io.StringIO()):
#         coco_eval.summarize()

#     # Final summary
#     final_ap_50_95 = coco_eval.stats[0]
#     final_ap_50 = coco_eval.stats[1]
#     final_ar_100_all = coco_eval.stats[8]
#     final_ar_100_large = coco_eval.stats[10]

#     summary = Table(title="📊 Final COCO Summary")
#     summary.add_column("Metric", justify="left")
#     summary.add_column("Value", justify="center")
#     summary.add_row("mAP@[IoU=0.50:0.95]", f"{final_ap_50_95:.4f}")
#     summary.add_row("mAP@0.5", f"{final_ap_50:.4f}")
#     summary.add_row("AR@100 (all)", f"{final_ar_100_all:.4f}")
#     summary.add_row("AR@100 (large)", f"{final_ar_100_large:.4f}")
#     console.print(summary)

#     # Per-class AP
#     class_aps = {}
#     for i in range(1, 5):
#         coco_eval.params.catIds = [i]
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         with contextlib.redirect_stdout(io.StringIO()):
#             coco_eval.summarize()
#         class_aps[dataset.id_to_name[i]] = coco_eval.stats[1]

#     class_table = Table(title="📊 Per-Class AP@0.5")
#     class_table.add_column("Class")
#     class_table.add_column("AP@0.5")
#     for cls, ap in class_aps.items():
#         class_table.add_row(cls, f"{ap:.4f}")
#     console.print(class_table)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_dir", required=True)
#     parser.add_argument("--ann_dir", required=True)
#     parser.add_argument("--weights", required=True)
#     args = parser.parse_args()
#     main(args)


#----------------------------------------------------------------------------------------------------------
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

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rich.console import Console
from rich.table import Table
from rich.progress import track

import io
import contextlib

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

# ------------------ Soft-NMS placeholder ------------------
from torchvision.ops import nms
def soft_nms_pytorch(boxes, scores, iou_threshold=0.5):
    return nms(boxes, scores, iou_threshold)

# ------------------ Model ------------------
def get_model(num_classes):
    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    model = retinanet_resnet50_fpn_v2(weights=weights)
    in_feat = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(in_feat, num_anchors, num_classes)
    return model

# ------------------ Main ------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.rule("[bold cyan]RetinaNet Inference")

    dataset = RSODDataset(args.dataset_dir, args.ann_dir, transforms=get_val_transform())
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_classes=5)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    gc.collect()
    torch.cuda.empty_cache()

    results = []
    
    # Create predictions.txt and write header
    with open("predictions.txt", "w") as pred_txt:
        pred_txt.write("image_name,image_id,true_labels,pred_label,score\n")
        
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(track(val_loader, description="🔍 Running inference...")):
                imgs = [img.to(device) for img in imgs]
                outputs = model(imgs)
                for j, output in enumerate(outputs):
                    boxes = output['boxes'].cpu()
                    scores = output['scores'].cpu()
                    labels = output['labels'].cpu()
                    keep = soft_nms_pytorch(boxes, scores)
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                    true_labels = targets[j]['labels'].cpu().tolist()
                    true_label_names = [dataset.id_to_name.get(l, "NA") for l in true_labels]
                    true_str = '/'.join(true_label_names) if true_label_names else "NA"

                    img_index = idx * val_loader.batch_size + j
                    img_name = os.path.basename(dataset.img_files[img_index])

                    for b, s, l in zip(boxes, scores, labels):
                        pred_name = dataset.id_to_name.get(int(l), "NA")
                        pred_txt.write(f"{img_name},{img_index},{true_str},{pred_name},{s:.4f}\n")

                        results.append({
                            "image_id": img_index,
                            "category_id": int(l),
                            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                            "score": float(s)
                        })

    with open("retina_predictions.json", "w") as f:
        json.dump(results, f)

    # Build COCO GT annotations
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {"year": 2025},
        "licenses": [],
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
    coco_eval.evaluate()
    coco_eval.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()

    # Final summary
    final_ap_50_95 = coco_eval.stats[0]
    final_ap_50 = coco_eval.stats[1]
    final_ar_100_all = coco_eval.stats[8]
    final_ar_100_large = coco_eval.stats[10]

    summary = Table(title="📊 Final COCO Summary")
    summary.add_column("Metric", justify="left")
    summary.add_column("Value", justify="center")
    summary.add_row("mAP@[IoU=0.50:0.95]", f"{final_ap_50_95:.4f}")
    summary.add_row("mAP@0.5", f"{final_ap_50:.4f}")
    summary.add_row("AR@100 (all)", f"{final_ar_100_all:.4f}")
    summary.add_row("AR@100 (large)", f"{final_ar_100_large:.4f}")
    console.print(summary)

    # Per-class AP
    class_aps = {}
    for i in range(1, 5):
        coco_eval.params.catIds = [i]
        coco_eval.evaluate()
        coco_eval.accumulate()
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.summarize()
        class_aps[dataset.id_to_name[i]] = coco_eval.stats[1]

    class_table = Table(title="📊 Per-Class AP@0.5")
    class_table.add_column("Class")
    class_table.add_column("AP@0.5")
    for cls, ap in class_aps.items():
        class_table.add_row(cls, f"{ap:.4f}")
    console.print(class_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--ann_dir", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    main(args)
