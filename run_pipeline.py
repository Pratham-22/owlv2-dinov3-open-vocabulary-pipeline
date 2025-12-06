import os
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoModel

from pipeline.config import Config
from pipeline.detection import OwlDetector
from pipeline.embeddings import ImageEmbeddingState
from pipeline.classification import classify_objects
from pipeline.utils import (
    load_image,
    find_images,
    ensure_dir,
    save_json,
    apply_nms,
)


def load_class_states(cfg, dino_model):
    class_dir = Path(cfg.CLASS_IMAGE_DIR)
    image_files = find_images(class_dir)

    if len(image_files) != 2:
        raise ValueError(
            f"Expected exactly 2 class images in {class_dir}, found {len(image_files)}"
        )

    class_states = {}
    print(f"[INFO] Loading class images from: {class_dir}")
    for img_path in image_files:
        class_name = img_path.stem
        print(f"  - {class_name}: {img_path}")
        img = load_image(img_path)
        state = ImageEmbeddingState(img, dino_model, cfg.DEVICE, cfg.PATCH_SIZE)
        class_states[class_name] = state

    return class_states


def save_detailed_log(cfg, detailed_log):
    ensure_dir(cfg.OUTPUT_DIR)
    log_path = os.path.join(cfg.OUTPUT_DIR, "owlv2_dino_batch_detailed_log.txt")

    with open(log_path, "w") as f:
        f.write("OWLv2 + DINOv3 Batch Processing - Detailed Log\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total objects processed: {len(detailed_log)}\n\n")

        for i, entry in enumerate(detailed_log):
            f.write(f"Object {i+1}:\n")
            for k, v in entry.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        # Summary stats
        class_counts = {}
        scores = []
        latencies = []

        for e in detailed_log:
            cls = e["predicted_label"]
            class_counts[cls] = class_counts.get(cls, 0) + 1
            scores.append(e["dino_confidence"])
            latencies.append(e["latency_ms"])

        f.write("Summary Statistics\n")
        f.write("-" * 80 + "\n")
        f.write("Class distribution:\n")
        for cls, c in class_counts.items():
            f.write(f"  {cls}: {c}\n")

        if scores:
            f.write(f"\nAvg confidence: {np.mean(scores):.4f}\n")
            f.write(f"Min confidence: {np.min(scores):.4f}\n")
            f.write(f"Max confidence: {np.max(scores):.4f}\n")

        if latencies:
            f.write(f"\nAvg latency: {np.mean(latencies):.2f} ms\n")
            f.write(f"Min latency: {np.min(latencies):.2f} ms\n")
            f.write(f"Max latency: {np.max(latencies):.2f} ms\n")

    print(f"[INFO] Detailed log written to: {log_path}")


def main():
    cfg = Config()
    ensure_dir(cfg.OUTPUT_DIR)
    cropped_dir = os.path.join(cfg.OUTPUT_DIR, "cropped_patches")
    if cfg.SAVE_CROPPED:
        ensure_dir(cropped_dir)

    print("=== OWLv2 + DINOv3 Open-Vocabulary Pipeline ===")
    print(f"Device: {cfg.DEVICE}")
    print(f"Target images: {cfg.TARGET_IMAGE_DIR}")
    print(f"Class images: {cfg.CLASS_IMAGE_DIR}")
    print(f"Output dir: {cfg.OUTPUT_DIR}\n")

    # Load models
    print("[INFO] Loading OWLv2...")
    detector = OwlDetector(cfg)

    print("[INFO] Loading DINOv3...")
    dino_model = AutoModel.from_pretrained(cfg.DINO_MODEL_ID).to(cfg.DEVICE)
    dino_model.eval()

    # Load class states
    class_states = load_class_states(cfg, dino_model)

    # Discover target images
    target_paths = find_images(cfg.TARGET_IMAGE_DIR)
    print(f"[INFO] Found {len(target_paths)} target images.\n")

    results = []
    detailed_log = []

    start_total = time.time()

    # Process in batches
    for batch_start in range(0, len(target_paths), cfg.BATCH_SIZE):
        batch_paths = target_paths[batch_start : batch_start + cfg.BATCH_SIZE]
        batch_imgs = []
        batch_states = []
        batch_names = []

        print(
            f"[INFO] Batch {batch_start//cfg.BATCH_SIZE + 1} "
            f"({len(batch_paths)} images)"
        )

        # Load + embed
        for p in batch_paths:
            try:
                img = load_image(p)
                state = ImageEmbeddingState(img, dino_model, cfg.DEVICE, cfg.PATCH_SIZE)
                batch_imgs.append(img)
                batch_states.append(state)
                batch_names.append(p.name)
            except Exception as e:
                print(f"[WARN] Skipping {p}: {e}")

        if not batch_imgs:
            continue

        # Detect
        t0 = time.time()
        detections_per_image = detector.detect_batch(batch_imgs)
        t_detection = time.time() - t0

        # NMS + prepare objects
        objects = []
        for img_idx, dets in enumerate(detections_per_image):
            dets_nms = apply_nms(dets, cfg.NMS_THRESHOLD)
            for obj_idx, d in enumerate(dets_nms):
                objects.append(
                    (
                        img_idx,
                        obj_idx,
                        d["bbox"],
                        d["confidence"],
                        d["prompt"],
                    )
                )

        if not objects:
            print("[INFO] No objects in this batch after NMS.\n")
            continue

        # Classify
        t1 = time.time()
        classified = classify_objects(
            objects,
            batch_states,
            class_states,
            max_workers=cfg.MAX_WORKERS,
        )
        t_class = time.time() - t1

        total_batch_time = t_detection + t_class
        avg_obj_time = total_batch_time / max(len(classified), 1)

        # Collect outputs
        for (
            img_idx,
            obj_idx,
            label,
            score,
            bbox,
            det_conf,
            prompt,
        ) in classified:
            img_name = batch_names[img_idx]
            x1, y1, x2, y2 = map(int, bbox)

            cropped_filename = None
            if cfg.SAVE_CROPPED:
                crop = batch_imgs[img_idx].crop((x1, y1, x2, y2))
                cropped_filename = (
                    f"{Path(img_name).stem}_obj{obj_idx}_{x1}_{y1}_{x2}_{y2}.jpg"
                )
                crop.save(os.path.join(cropped_dir, cropped_filename))

            obj_record = {
                "img": img_name,
                "prediction": 1,
                "label": label,
                "latency": round(avg_obj_time * 1000, 2),
                "confidence": float(score),
                "bounding_box": [float(v) for v in bbox],
            }
            results.append(obj_record)

            detailed_log.append(
                {
                    "image": img_name,
                    "object_id": obj_idx,
                    "bbox": [float(v) for v in bbox],
                    "owlv2_prompt": prompt,
                    "owlv2_confidence": float(det_conf),
                    "predicted_label": label,
                    "dino_confidence": float(score),
                    "latency_ms": round(avg_obj_time * 1000, 2),
                    "cropped_file": cropped_filename,
                }
            )

        print(
            f"[INFO] Batch done: {len(classified)} objects | "
            f"det {t_detection:.2f}s, cls {t_class:.2f}s\n"
        )

        if cfg.DEVICE == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - start_total

    # Save results
    results_path = os.path.join(cfg.OUTPUT_DIR, "owlv2_dino_batch_results.json")
    save_json(results_path, results)
    print(f"[INFO] Saved results to: {results_path}")

    if cfg.SAVE_DETAILED_LOG and detailed_log:
        save_detailed_log(cfg, detailed_log)

    print("\n=== Summary ===")
    print(f"Total images processed: {len(target_paths)}")
    print(f"Total objects: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    if results:
        print(f"Avg time / object: {total_time/len(results):.3f}s")


if __name__ == "__main__":
    main()
