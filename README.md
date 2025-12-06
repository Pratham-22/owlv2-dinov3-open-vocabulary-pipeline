# OWLv2 + DINOv3 Open-Vocabulary Detection & Patch-Similarity Classification

This repository implements a two-stage **open-vocabulary vision pipeline** that combines:

1. **OWLv2** for open-vocabulary object detection using text prompts  
2. **DINOv3** for patch-level feature embeddings and similarity-based classification

It is designed as a research-style, GPU-accelerated pipeline with batch and parallel processing.

---

##  Key Features

- **Open-vocabulary detection** with OWLv2 and natural language prompts  
- **Zero-shot classification** using DINOv3 patch embeddings and class images  
- **Batch processing** of images for faster inference  
- **Parallel classification** across detected objects  
- **Patch-level region similarity**, not just global embeddings  
- **Cropped patch saving**, JSON outputs, and detailed logs  
- Ready to plug into ecology / agriculture / plant monitoring workflows

---

##  Pipeline Overview

```text
Input Images ──► OWLv2 Detection ──► Filter + NMS ──► BBox Regions
                                                   │
                                                   ▼
                           DINOv3 Patch Embeddings (image + class images)
                                                   │
                                                   ▼
                             Patch Similarity + Argmax over Classes
                                                   │
                                                   ▼
                                Final Labels + Crops + Metrics
