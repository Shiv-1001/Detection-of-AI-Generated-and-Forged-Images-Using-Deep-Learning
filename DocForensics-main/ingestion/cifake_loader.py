"""
CIFAKE dataset integration.

CIFAKE ("Real and AI-Generated Synthetic Images"):
  Kaggle:  https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
  Source paper: Bird, J.J. and Lotfi, A. (2024). "CIFAKE: Image Classification and
  Explainable Identification of AI-Generated Synthetic Images." IEEE Access.

Dataset facts (verified against the dataset's own documentation, not assumed):
  - 120,000 images total, class-balanced: 60,000 REAL (drawn from CIFAR-10) and
    60,000 FAKE (generated with Stable Diffusion v1.4).
  - Canonical split: 100,000 for training (50,000 REAL / 50,000 FAKE) and
    20,000 for testing (10,000 REAL / 10,000 FAKE).
  - Images are 32x32 RGB.

This module downloads CIFAKE and folds it into this project's existing 3-class
layout (data/original, data/aigenerated, data/forged -> labels 0/1/2 in
model/dataset.py) so it plugs straight into TamperDataset / TamperNet without
touching the rest of the pipeline:
  - CIFAKE "REAL"  -> data/original    (label 0)
  - CIFAKE "FAKE"  -> data/aigenerated (label 1)
  - data/forged (label 2, copy-move/splice) is untouched — CIFAKE has no
    localization masks, so it cannot supply that class.

Because CIFAKE images are small (32x32), they are upsampled to MODEL_INPUT_SIZE
by the existing TamperDataset resize step; this is a real quality trade-off
worth noting in the report (see docs).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from core.config import AI_GENERATED_DIR, ORIGINAL_DIR

KAGGLE_DATASET_SLUG = "birdy654/cifake-real-and-ai-generated-synthetic-images"


def download_cifake(dest_dir: Path | None = None) -> Path:
    """
    Download CIFAKE from Kaggle via kagglehub.

    Requires Kaggle API credentials to be available in the environment
    (either ~/.kaggle/kaggle.json, or the KAGGLE_USERNAME / KAGGLE_KEY
    environment variables). This function does not fabricate a dataset —
    if credentials are missing it raises, rather than silently continuing.
    """
    try:
        import kagglehub
    except ImportError as e:
        raise ImportError(
            "kagglehub is required to download CIFAKE. Install with:\n"
            "    pip install kagglehub"
        ) from e

    path = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    print(f"CIFAKE downloaded/cached at: {path}")
    return path


def _find_split_dirs(cifake_root: Path) -> dict:
    """
    CIFAKE ships as train/{REAL,FAKE} and test/{REAL,FAKE}. Kaggle's exact
    top-level folder naming has varied historically, so search rather than
    hard-code a single layout.
    """
    found = {}
    for split in ("train", "test"):
        for cls in ("REAL", "FAKE"):
            matches = list(cifake_root.rglob(f"{split}/{cls}"))
            if matches:
                found[(split, cls)] = matches[0]
    return found


def populate_from_cifake(
    cifake_root: Path,
    max_per_class: int | None = None,
    splits: tuple[str, ...] = ("train", "test"),
    link: bool = True,
) -> dict:
    """
    Copy (or symlink) CIFAKE images into data/original and data/aigenerated.

    max_per_class: cap the number of images pulled per (split, class) pair.
        Useful for a quick smoke run before committing to the full 120k
        images. Pass None to use everything available.
    link: use symlinks instead of copies (fast, no extra disk usage). Falls
        back to copying automatically on filesystems that reject symlinks
        (e.g. some Windows setups without developer mode).

    Returns a dict of the number of images placed per destination class, so
    the caller can print/verify the resulting class balance rather than
    assuming it.
    """
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    AI_GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    split_dirs = _find_split_dirs(cifake_root)
    if not split_dirs:
        raise FileNotFoundError(
            f"Could not locate train/REAL, train/FAKE, test/REAL, test/FAKE "
            f"under {cifake_root}. Inspect the downloaded folder structure "
            f"and adjust _find_split_dirs() if Kaggle has changed the layout."
        )

    counts = {"original": 0, "aigenerated": 0}
    dest_map = {"REAL": (ORIGINAL_DIR, "original"), "FAKE": (AI_GENERATED_DIR, "aigenerated")}

    for (split, cls), src_dir in split_dirs.items():
        if split not in splits:
            continue
        dest_dir, dest_key = dest_map[cls]
        files = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
        if max_per_class is not None:
            files = files[:max_per_class]

        for f in files:
            dest_name = f"cifake_{split}_{cls.lower()}_{f.name}"
            dest_path = dest_dir / dest_name
            if dest_path.exists():
                continue
            if link:
                try:
                    dest_path.symlink_to(f.resolve())
                except OSError:
                    shutil.copy2(f, dest_path)
            else:
                shutil.copy2(f, dest_path)
            counts[dest_key] += 1

    print(f"CIFAKE ingestion complete: original(+{counts['original']}), "
          f"aigenerated(+{counts['aigenerated']})")
    return counts


def report_class_balance() -> dict:
    """Count what's actually on disk right now — never assumed, always measured."""
    from core.config import FORGED_DIR
    exts = ("*.jpg", "*.jpeg", "*.png")
    def _count(d: Path) -> int:
        return sum(1 for e in exts for _ in d.glob(e))
    counts = {
        "original":    _count(ORIGINAL_DIR),
        "aigenerated": _count(AI_GENERATED_DIR),
        "forged":      _count(FORGED_DIR),
    }
    total = sum(counts.values()) or 1
    for k, v in counts.items():
        print(f"  {k:12s}: {v:6d}  ({100*v/total:5.1f}%)")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CIFAKE and fold it into data/")
    parser.add_argument("--max-per-class", type=int, default=None,
                         help="Cap images per (split, class) — omit for the full 120k.")
    parser.add_argument("--no-link", action="store_true",
                         help="Copy files instead of symlinking.")
    args = parser.parse_args()

    root = download_cifake()
    populate_from_cifake(root, max_per_class=args.max_per_class, link=not args.no_link)
    report_class_balance()
