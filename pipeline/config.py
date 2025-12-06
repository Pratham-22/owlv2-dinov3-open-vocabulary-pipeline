import torch
import os


class Config:
    # Model IDs (Hugging Face)
    OWL_MODEL_ID = "google/owlv2-base-patch16-ensemble"
    DINO_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    # Detection settings
    DETECTION_PROMPTS = [["a photo of a plant", "a plant", "vegetation", "foliage"]]
    CONFIDENCE_THRESHOLD = 0.1
    NMS_THRESHOLD = 0.5

    # Batch / parallelism
    BATCH_SIZE = 4
    OBJECT_BATCH_SIZE = 8
    MAX_WORKERS = 4

    # Patch / device
    PATCH_SIZE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths (relative to repo root)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CLASS_IMAGE_DIR = os.path.join(ROOT_DIR, "examples", "class_images")
    TARGET_IMAGE_DIR = os.path.join(ROOT_DIR, "examples", "test_images")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "results")

    # Output options
    SAVE_CROPPED = True
    SAVE_DETAILED_LOG = True
