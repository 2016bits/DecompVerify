"""Cross-encoder reranker for fact-level candidate sentences.

A lightweight wrapper around a (query, passage) -> relevance-score model.
The default is BGE-reranker-v2-m3 (XLMRobertaForSequenceClassification),
which lives on disk at ``/mnt/data/hezhisheng/models/bge-reranker-v2-m3``.
The wrapper:

* loads lazily — instantiating the class does no IO, the model only hits
  disk on the first ``score`` call (so an unused reranker costs nothing);
* batches pairs on the configured device (``cuda`` by default if
  available, else ``cpu``);
* applies sigmoid to the raw logit so downstream code sees a [0, 1]
  score that's drop-in comparable to the legacy heuristic ``ce_score``.

On any load / inference failure the score call returns an empty array
and the caller (``retriever._retrieve_for_fact``) falls back to the
heuristic path — the pipeline never crashes because of the reranker.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


DEFAULT_RERANKER_PATH = "/mnt/data/hezhisheng/models/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """Thin (query, passage) -> sigmoid score wrapper.

    The constructor only stores config. The model is loaded on the first
    ``score`` call (``_ensure_loaded``) so importing this module never
    forces torch / transformers to initialise.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_RERANKER_PATH,
        max_length: int = 256,
        device: Optional[str] = None,
        batch_size: int = 64,
        use_fp16: bool = True,
    ):
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self._device = device
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._load_error: Optional[str] = None

    # ------------------------------------------------------------- loading
    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
            device = self._device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            )
            model.eval()
            if device.startswith("cuda") and self.use_fp16:
                model = model.half()
            model.to(device)
            self._tokenizer = tokenizer
            self._model = model
            self._torch = torch
            self._device = device
            return True
        except Exception as exc:  # pragma: no cover - defensive
            self._load_error = f"{type(exc).__name__}: {exc}"
            print(f"[reranker] failed to load {self.model_path}: "
                  f"{self._load_error}", flush=True)
            return False

    # ---------------------------------------------------------- prediction
    def score(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Score a list of ``(query, passage)`` pairs.

        Returns a float32 numpy array of sigmoid scores in ``[0, 1]``. On
        any failure (model not available, OOM, …) returns ``np.empty(0)``
        and the caller is expected to fall back to the heuristic path.
        """
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        if not self._ensure_loaded():
            return np.zeros(0, dtype=np.float32)

        torch = self._torch
        try:
            all_scores: List[np.ndarray] = []
            for i in range(0, len(pairs), self.batch_size):
                chunk = pairs[i:i + self.batch_size]
                queries = [p[0] or "" for p in chunk]
                passages = [p[1] or "" for p in chunk]
                inputs = self._tokenizer(
                    queries, passages,
                    padding=True, truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)
                with torch.no_grad():
                    logits = self._model(**inputs).logits
                    # BGE-reranker emits a single logit per pair.
                    logits = logits.view(-1).float()
                    scores = torch.sigmoid(logits).cpu().numpy()
                all_scores.append(scores.astype(np.float32))
            return np.concatenate(all_scores) if all_scores else np.zeros(
                0, dtype=np.float32)
        except Exception as exc:
            print(f"[reranker] scoring failed: {type(exc).__name__}: {exc}",
                  flush=True)
            return np.zeros(0, dtype=np.float32)


# Process-wide cache so multi-claim runs share one loaded model.
_RERANKER_CACHE: dict = {}


def get_reranker(
    model_path: str,
    device: Optional[str] = None,
    batch_size: int = 64,
    max_length: int = 256,
    use_fp16: bool = True,
) -> Optional[CrossEncoderReranker]:
    """Return a cached reranker for ``model_path`` (or None to disable).

    Passing an empty / falsy ``model_path`` returns ``None`` so callers can
    use ``--reranker_path ""`` as a clean "disable" toggle on the CLI.
    """
    if not model_path:
        return None
    key = (model_path, device, batch_size, max_length, bool(use_fp16))
    if key not in _RERANKER_CACHE:
        r = CrossEncoderReranker(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            use_fp16=use_fp16,
        )
        # Eager-load so we fail fast at startup, not after 50 claims.
        if not r._ensure_loaded():
            _RERANKER_CACHE[key] = None
        else:
            _RERANKER_CACHE[key] = r
    return _RERANKER_CACHE[key]
