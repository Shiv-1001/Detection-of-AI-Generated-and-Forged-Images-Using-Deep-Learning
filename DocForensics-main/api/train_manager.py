"""Runs model training in a background thread and exposes status for the API.

Mirrors the handwritten workflow's training stage: dataset (Original /
Forged) -> Model input -> TamperNet (4x downsample encoder, L1 decoder
skips, 1 classification head) -> checkpoints/best.pt.
"""
import threading
import time

from core.config import (AI_GENERATED_DIR, CHECKPOINTS_DIR, FORGED_DIR,
                         MAX_EPOCHS, ORIGINAL_DIR)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}

_lock = threading.Lock()
_thread: threading.Thread | None = None
_state = {
    'status':        'idle',   # idle | running | done | error
    'current_epoch': 0,
    'total_epochs':  MAX_EPOCHS,
    'history':       [],
    'error':         None,
}


def dataset_counts() -> dict:
    def _count(d):
        if not d.exists():
            return 0
        return sum(1 for path in d.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return {
        'ai_generated': _count(AI_GENERATED_DIR),
        'forged': _count(FORGED_DIR),
        'original': _count(ORIGINAL_DIR),
    }


def has_checkpoint() -> bool:
    return (CHECKPOINTS_DIR / 'best.pt').exists()


def get_status() -> dict:
    with _lock:
        return {**_state, 'history': list(_state['history'])}


def _on_epoch(epoch: int, metrics: dict):
    with _lock:
        _state['current_epoch'] = epoch
        _state['history'].append({'epoch': epoch, **metrics})


def _run():
    from model.train import train as run_train

    with _lock:
        _state.update(status='running', current_epoch=0, total_epochs=MAX_EPOCHS,
                       history=[], error=None)
    try:
        run_train(on_epoch=_on_epoch)
        with _lock:
            _state['status'] = 'done'
    except Exception as e:
        with _lock:
            _state['status'] = 'error'
            _state['error']  = str(e)


def start_training() -> bool:
    """Returns False if a training run is already in progress."""
    global _thread
    with _lock:
        if _state['status'] == 'running':
            return False
    _thread = threading.Thread(target=_run, daemon=True)
    _thread.start()
    # give it a moment so /train/status right after reflects 'running'
    time.sleep(0.05)
    return True
