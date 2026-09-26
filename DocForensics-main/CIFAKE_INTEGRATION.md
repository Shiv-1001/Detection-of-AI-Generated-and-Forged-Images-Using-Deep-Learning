# CIFAKE Integration — what changed and how to run it

## What was added / changed

1. **`ingestion/cifake_loader.py`** (new) — downloads the CIFAKE dataset
   (https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
   via `kagglehub`, and folds it into the project's existing 3-class layout:
   - CIFAKE `REAL` → `data/original` (label 0)
   - CIFAKE `FAKE` → `data/aigenerated` (label 1)
   - `data/forged` (label 2) is untouched — CIFAKE has no localization masks,
     so it cannot supply the forged/copy-move class.

2. **`model/evaluate.py`** (updated) — `evaluate()` now also returns
   `accuracy`, `precision`, `recall`, and a `confusion_matrix`, in addition to
   the existing `auc`, `f1`, `pixel_iou`.

3. **`model/train.py`** (updated) — epoch log line now prints
   accuracy/precision/recall alongside loss/AUC/IoU.

4. **`model/plots.py`** (new) — plots training-loss / validation-AUC curves
   and a confusion-matrix heatmap from `model/checkpoints/history.json` (the
   file `train.py` already writes). Run this *after* training.

5. **`requirements.txt`** — added `kagglehub` and `matplotlib`.

## How to actually run it

I was not able to execute any of this in this session: the sandbox has no
network access to kaggle.com (only PyPI/npm/GitHub are reachable), and even
with the dataset present, training a CNN on 100k+ images needs a GPU and
realistically tens of minutes to hours — well beyond what a single chat
turn/tool call can do. So the steps below are what **you** run (locally, in
Colab, or in a Kaggle notebook — the last one is the path of least
resistance since the dataset is already local there):

```bash
# 1. Kaggle credentials (skip this step if you're already in a Kaggle notebook)
#    Get kaggle.json from kaggle.com/settings -> Create New Token
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 2. Install the new deps
pip install -r requirements.txt

# 3. Download CIFAKE and populate data/original + data/aigenerated
#    (--max-per-class 2000 for a quick smoke test; omit it for the full 120k)
python -m ingestion.cifake_loader --max-per-class 2000

# 4. Train (uses the existing TamperNet / TamperDataset pipeline unchanged)
python -m model.train

# 5. Plot training curves + confusion matrix from the run you just did
python -m model.plots
```

`model/train.py` already writes per-epoch metrics (now including accuracy,
precision, recall) to `model/checkpoints/history.json`, and saves the best
checkpoint to `model/checkpoints/best.pt`. **Report the numbers from your own
`history.json` after running this** — do not reuse the reference-paper or
literature numbers in `CIFAKE_INTEGRATION.md`/the report as if they were this
run's result; they are cited comparisons, not this project's measured output.

## Known trade-off to be aware of

CIFAKE images are natively 32x32 px. `TamperDataset` resizes everything to
`MODEL_INPUT_SIZE` (128 px by default, in `core/config.py`), so CIFAKE images
get upsampled — this cannot add detail that was never captured, and is worth
noting as a limitation in any report of results, not as a hidden gotcha.
