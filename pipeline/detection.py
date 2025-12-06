import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection


class OwlDetector:
    """
    Wrapper around OWLv2 for batched open-vocabulary object detection.
    """

    def __init__(self, config):
        self.cfg = config
        self.processor = Owlv2Processor.from_pretrained(self.cfg.OWL_MODEL_ID)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.cfg.OWL_MODEL_ID
        ).to(self.cfg.DEVICE)
        self.model.eval()

    def detect_batch(self, images):
        """
        images: list of PIL.Images
        returns: list[list[dict]] per image: {"bbox", "confidence", "prompt", "label_idx"}
        """
        inputs = self.processor(
            text=self.cfg.DETECTION_PROMPTS,
            images=images,
            return_tensors="pt"
        ).to(self.cfg.DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [img.size[::-1] for img in images], device=self.cfg.DEVICE
        )

        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.cfg.CONFIDENCE_THRESHOLD,
        )

        parsed = []
        prompts = self.cfg.DETECTION_PROMPTS[0]

        for res in results:
            boxes = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            labels = res["labels"].cpu().numpy()

            img_dets = []
            for b, s, l in zip(boxes, scores, labels):
                prompt = prompts[int(l) % len(prompts)]
                img_dets.append(
                    {
                        "bbox": b.tolist(),
                        "confidence": float(s),
                        "prompt": prompt,
                        "label_idx": int(l),
                    }
                )
            parsed.append(img_dets)

        return parsed
