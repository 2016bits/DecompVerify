"""Mean-pooled sentence encoder backed by HuggingFace ``transformers``.

We deliberately avoid the ``sentence_transformers`` runtime because the
project's pinned ``transformers==4.32`` is incompatible with the latest
``sentence_transformers`` release. The local MiniLM checkpoint at
``/mnt/data/public/LLM/all-MiniLM-L6-v2`` ships a standard BERT-style
model, so mean-pooled embeddings reproduce the original sentence
encoder exactly.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL_PATH = "/data/public/LLM/all-MiniLM-L6-v2"


class SentenceEncoder:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 64,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.model.eval()
        with torch.no_grad():
            test = self.tokenizer(["x"], return_tensors="pt").to(device)
            self.dim = self.model(**test).last_hidden_state.shape[-1]

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t else " " for t in texts[i : i + self.batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hs = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)


_GLOBAL_ENCODER: SentenceEncoder | None = None


def get_encoder(model_path: str = DEFAULT_MODEL_PATH) -> SentenceEncoder:
    """Singleton accessor so we only pay the model-load cost once per process."""
    global _GLOBAL_ENCODER
    if _GLOBAL_ENCODER is None or _GLOBAL_ENCODER.model_path != model_path:
        _GLOBAL_ENCODER = SentenceEncoder(model_path)
    return _GLOBAL_ENCODER
