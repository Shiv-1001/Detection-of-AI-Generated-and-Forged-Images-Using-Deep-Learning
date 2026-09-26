import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from core.config import (AI_GENERATED_DIR, FORGED_DIR, MASKS_DIR,
                         MODEL_INPUT_SIZE, ORIGINAL_DIR)

IMAGE_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif')


def _images(directory):
    return sorted(path for pattern in IMAGE_EXTS for path in directory.glob(pattern))


class TamperDataset(Dataset):
    def __init__(self, split: str = 'train', val_fraction: float = 0.15):
        original = _images(ORIGINAL_DIR)
        ai_generated = _images(AI_GENERATED_DIR)
        forged = _images(FORGED_DIR)

        all_items = ([(p, 0) for p in original] +
                 [(p, 1) for p in ai_generated] +
                 [(p, 2) for p in forged])

        rng = np.random.default_rng(42)
        order = rng.permutation(len(all_items)).tolist()
        all_items = [all_items[i] for i in order]

        cut = int(len(all_items) * (1 - val_fraction))
        self.items   = all_items[:cut] if split == 'train' else all_items[cut:]
        self.size    = MODEL_INPUT_SIZE
        self.augment = (split == 'train')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i) -> tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.items[i]

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0

        if label == 2:
            mask_path = MASKS_DIR / (path.stem + '.png')
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.size, self.size))
                mask = (mask > 127).astype(np.float32)
            else:
                mask = np.ones((self.size, self.size), dtype=np.float32)
        else:
            mask = np.zeros((self.size, self.size), dtype=np.float32)

        if self.augment:
            img, mask = _augment(img, mask)

        img_t  = torch.from_numpy(img).permute(2, 0, 1)   # (3, H, W)
        mask_t = torch.from_numpy(mask).unsqueeze(0)       # (1, H, W)
        return img_t, mask_t, label


def _augment(img: np.ndarray, mask: np.ndarray):
    size = img.shape[0]

    if np.random.rand() > 0.5:
        img  = img[:, ::-1, :].copy()
        mask = mask[:, ::-1].copy()
    if np.random.rand() > 0.5:
        img  = img[::-1, :, :].copy()
        mask = mask[::-1, :].copy()

    if np.random.rand() > 0.5:
        alpha = np.random.uniform(0.7, 1.3)
        beta  = np.random.uniform(-0.1, 0.1)
        img   = np.clip(img * alpha + beta, 0, 1).astype(np.float32)

    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
        img   = np.clip(img + noise, 0, 1)

    # Resolution-jitter: randomly shrink then re-enlarge, on ALL classes
    # regardless of label. Without this, "looks upscaled from something tiny"
    # can end up perfectly correlated with one class (e.g. because that
    # class's source images were natively much smaller), letting the model
    # learn that as a shortcut instead of real tamper/generation evidence.
    # Applying it uniformly across every class breaks that correlation.
    if np.random.rand() > 0.5:
        shrink_to = np.random.randint(max(8, size // 8), size)
        small = cv2.resize(img, (shrink_to, shrink_to), interpolation=cv2.INTER_AREA)
        img   = cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)

    return img, mask
