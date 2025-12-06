import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import math


class ImageEmbeddingState:
    """
    Holds DINOv3 patch embeddings + geometry for one image.
    """

    def __init__(self, pil_img, model, device, patch_size):
        self.pil = pil_img
        self.ps = patch_size

        padded, _ = self._pad_to_multiple(pil_img, patch_size)
        pixel_values = self._preprocess(padded).to(device)
        _, _, H, W = pixel_values.shape

        self.rows = H // patch_size
        self.cols = W // patch_size

        with torch.no_grad():
            out = model(pixel_values=pixel_values)
        hs = out.last_hidden_state.squeeze(0).cpu().numpy()

        T, D = hs.shape
        n_patches = self.rows * self.cols
        n_special = T - n_patches  # cls / register tokens

        if n_special < 1:
            raise RuntimeError(
                f"Token shape mismatch: T={T}, rows*cols={n_patches}"
            )

        patch_tokens = hs[n_special:, :].reshape(self.rows, self.cols, D)
        self.X = patch_tokens.reshape(-1, D)
        self.Xn = self.X / (np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-8)

    @staticmethod
    def _preprocess(img):
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return tfm(img).unsqueeze(0)

    @staticmethod
    def _pad_to_multiple(img, multiple):
        W, H = img.size
        H_pad = math.ceil(H / multiple) * multiple
        W_pad = math.ceil(W / multiple) * multiple

        if H_pad == H and W_pad == W:
            return img, None

        canvas = Image.new("RGB", (W_pad, H_pad), (0, 0, 0))
        canvas.paste(img, (0, 0))
        return canvas, (0, 0, W_pad - W, H_pad - H)
