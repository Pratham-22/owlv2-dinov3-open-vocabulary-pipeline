from pathlib import Path
from PIL import Image
import json
import os
import torch
from torchvision.ops import nms


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG")


def load_image(path):
    return Image.open(path).convert("RGB")


def find_images(root_dir):
    root = Path(root_dir)
    return sorted([p for p in root.rglob("*") if p.suffix in IMAGE_EXTS])


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def apply_nms(dets, iou_thresh):
    if not dets:
        return dets

    boxes = torch.tensor([d["bbox"] for d in dets], dtype=torch.float32)
    scores = torch.tensor([d["confidence"] for d in dets], dtype=torch.float32)

    keep = nms(boxes, scores, iou_thresh)
    keep = keep.tolist()
    return [dets[i] for i in keep]
