import cv2
import numpy as np
from scipy.spatial import cKDTree

from core.config import CM_BLOCK_SIZE, CM_MATCH_THRESH, CM_STRIDE
from core.types import BBox, Detection
from detectors.base import Context, Detector

CM_MIN_AC_ENERGY = 40.0
CM_MIN_CLUSTER = 10


def block_hash_features(img: np.ndarray,
                        block: int = CM_BLOCK_SIZE,
                        stride: int = CM_STRIDE) -> list[tuple[np.ndarray, int, int]]:
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    feats = []
    for y in range(0, h - block, stride):
        for x in range(0, w - block, stride):
            patch = gray[y:y+block, x:x+block]
            dct = cv2.dct(patch)
            sig = dct[:4, :4].copy().flatten()
            sig[0] = 0.0
            if np.linalg.norm(sig) < CM_MIN_AC_ENERGY:
                continue
            feats.append((sig, x, y))
    return feats


def find_shifted_duplicates(feats, block: int,
                            min_distance: int = 48) -> dict[tuple, list[BBox]]:
    """Group matching block pairs by their (dx, dy) shift vector.

    Uses a KD-tree over the DCT signatures instead of comparing every block
    against every other block (O(n log n) instead of O(n^2)) — the brute-force
    version took effectively forever on a full-resolution photo (tens of
    thousands of blocks -> ~1 billion pairwise comparisons).
    """
    shifts: dict[tuple, list[BBox]] = {}
    n = len(feats)
    if n < 2:
        return shifts

    sigs  = np.stack([f[0] for f in feats])
    norms = np.linalg.norm(sigs, axis=1)
    coords = np.array([[f[1], f[2]] for f in feats])

    tree = cKDTree(sigs)

    # Upper-bound radius covering the original relative-similarity threshold:
    # match requires 1 - |sig_i-sig_j| / (|sig_i|+|sig_j|) > CM_MATCH_THRESH
    # i.e. |sig_i-sig_j| < (1-CM_MATCH_THRESH) * (|sig_i|+|sig_j|).
    # Bounding (|sig_i|+|sig_j|) by 2*max(norms) gives a safe (possibly loose)
    # radius; the exact formula is re-applied below as a precise filter.
    max_norm = float(norms.max()) if n else 0.0
    radius = max((1 - CM_MATCH_THRESH) * 2 * max_norm, 1e-6)

    pairs = tree.query_pairs(r=radius, output_type='ndarray')
    for i, j in pairs:
        xi, yi = coords[i]
        xj, yj = coords[j]
        dx, dy = int(xj - xi), int(yj - yi)
        if (dx * dx + dy * dy) ** 0.5 < min_distance:
            continue
        denom = norms[i] + norms[j]
        if denom == 0:
            continue
        if 1 - np.linalg.norm(sigs[i] - sigs[j]) / denom > CM_MATCH_THRESH:
            key = (round(dx / 8) * 8, round(dy / 8) * 8)
            shifts.setdefault(key, []).extend(
                [BBox(int(xi), int(yi), block, block), BBox(int(xj), int(yj), block, block)]
            )
    return shifts


class CopyMoveDetector(Detector):
    name = 'copy_move'

    def run(self, img: np.ndarray, ctx: Context) -> Detection:
        try:
            feats  = block_hash_features(img)
            shifts = find_shifted_duplicates(feats, CM_BLOCK_SIZE)

            best_boxes = max(shifts.values(), key=len) if shifts else []
            cluster    = len(best_boxes) // 2

            density = 0.0
            if best_boxes:
                xs = [b.x for b in best_boxes]; ys = [b.y for b in best_boxes]
                bb_area = (max(xs) - min(xs) + CM_BLOCK_SIZE) * \
                          (max(ys) - min(ys) + CM_BLOCK_SIZE)
                blocks_area = len(best_boxes) * CM_BLOCK_SIZE * CM_BLOCK_SIZE
                density = float(blocks_area / (bb_area + 1e-6))

            flagged = cluster >= CM_MIN_CLUSTER and density >= 0.25
            score   = float(np.clip(cluster / 3000.0, 0, 0.6)) if flagged else 0.0

            heatmap = np.zeros(img.shape[:2], dtype=np.float32)
            regions = []
            if flagged:
                for b in best_boxes:
                    heatmap[b.y:b.y+b.h, b.x:b.x+b.w] = 1.0
                    regions.append(b)

            return Detection(
                detector_name=self.name,
                score=score,
                heatmap=heatmap,
                regions=regions,
                details={'cluster_size': cluster,
                         'density': round(density, 3),
                         'shift_groups': len(shifts)},
            )
        except Exception as e:
            d = self._empty()
            d.details['error'] = str(e)
            return d
