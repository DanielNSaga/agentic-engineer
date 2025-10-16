"""Lightweight embedding store used for similarity lookups in the index."""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


@dataclass(frozen=True)
class EmbeddingRecord:
    """Persistable embedding entry."""

    path: str
    vector: List[float]


class HashEmbeddingEncoder:
    """Deterministic placeholder encoder until a real model is available."""

    def __init__(self, dimension: int = 32) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, text: str) -> List[float]:
        if not text.strip():
            return [0.0] * self._dimension

        accumulator = [0.0] * self._dimension
        for token in text.split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(self._dimension):
                byte_value = digest[index % len(digest)]
                score = (byte_value / 255.0) * 2.0 - 1.0
                accumulator[index] += score

        norm = math.sqrt(sum(component * component for component in accumulator))
        if norm == 0:
            return [0.0] * self._dimension
        return [component / norm for component in accumulator]


class EmbeddingsIndex:
    """Lightweight FAISS-inspired index with a pure Python fallback."""

    def __init__(
        self,
        storage_path: Path,
        encoder: Optional[HashEmbeddingEncoder] = None,
    ) -> None:
        self._storage_path = storage_path
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._encoder = encoder or HashEmbeddingEncoder()
        self._records: Dict[str, EmbeddingRecord] = {}
        self._faiss_index: Optional[object] = None
        self._id_lookup: List[str] = []
        self._load()
        self._rebuild_backend()

    @property
    def is_empty(self) -> bool:
        return not self._records

    def index_document(self, path: Path, content: str) -> None:
        path_key = self._path_key(path)
        vector = self._encoder.encode(content)
        self._records[path_key] = EmbeddingRecord(path=path_key, vector=vector)
        self._persist()
        self._rebuild_backend()

    def remove(self, path: Path) -> None:
        path_key = self._path_key(path)
        if path_key in self._records:
            del self._records[path_key]
            self._persist()
            self._rebuild_backend()

    def search(self, query: str, limit: int = 5) -> List[Tuple[str, float]]:
        query_vector = self._encoder.encode(query)
        if not any(query_vector):
            return []
        if self._faiss_index is not None:
            return self._search_with_faiss(query_vector, limit)
        return self._search_fallback(query_vector, limit)

    def vectors(self) -> Dict[str, EmbeddingRecord]:
        return dict(self._records)

    def _search_fallback(self, query_vector: Sequence[float], limit: int) -> List[Tuple[str, float]]:
        scores: List[Tuple[str, float]] = []
        for record in self._records.values():
            score = sum(a * b for a, b in zip(query_vector, record.vector))
            if score > 0.0:
                scores.append((record.path, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:limit]

    def _search_with_faiss(self, query_vector: Sequence[float], limit: int) -> List[Tuple[str, float]]:
        assert faiss is not None  # pragma: no cover - guarded by constructor
        import numpy as np  # type: ignore

        query_array = np.array([query_vector], dtype="float32")
        scores, indices = self._faiss_index.search(query_array, limit)
        results: List[Tuple[str, float]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self._id_lookup):
                continue
            results.append((self._id_lookup[index], float(score)))
        return results

    def _load(self) -> None:
        if not self._storage_path.exists():
            return
        data = json.loads(self._storage_path.read_text(encoding="utf-8"))
        for path, payload in data.items():
            self._records[path] = EmbeddingRecord(path=path, vector=list(payload["vector"]))

    def _persist(self) -> None:
        payload = {
            path: asdict(record)
            for path, record in sorted(self._records.items())
        }
        self._storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _rebuild_backend(self) -> None:
        if faiss is None:
            self._faiss_index = None
            self._id_lookup = []
            return
        try:  # pragma: no cover - requires optional dependency
            import numpy as np  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            self._faiss_index = None
            self._id_lookup = []
            return

        if not self._records:
            self._faiss_index = None
            self._id_lookup = []
            return

        dimension = self._encoder.dimension
        index = faiss.IndexFlatIP(dimension)
        vectors = [record.vector for record in self._records.values()]
        matrix = np.array(vectors, dtype="float32")
        index.add(matrix)
        self._faiss_index = index
        self._id_lookup = list(self._records.keys())

    def _path_key(self, path: Path) -> str:
        return path.as_posix()
