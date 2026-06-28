"""Graceful persistence (D2 corpus). ALWAYS writes JSONL; additionally mirrors to
MongoDB / Atlas when MONGODB_URI (or MONGODB_ATLAS_URI) is set — never crashes the
run if Mongo is down.

Collections:
  personas — the validated persona population (with care vectors)
  episodes — every red-team episode (persona, request, action, gate, judge, reward, outcome)

Atlas upgrade path: add a Vector Search index over an embedding field on `episodes`
for novelty / nearest-prior-attack (left as a later, optional enhancement).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path

EMBED_DIM = 384  # all-MiniLM-L6-v2; matches the Atlas vector index below
_embedder = None


def _embed(text: str):
    """Sentence-Transformer embedding (lazy, graceful). Returns list[float] or None
    if sentence-transformers isn't installed — so corpus storage never hard-depends
    on it. Powers Atlas Vector Search (novelty / nearest-prior-attack)."""
    global _embedder
    if _embedder is False:
        return None
    try:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return _embedder.encode(text or "", normalize_embeddings=True).tolist()
    except Exception:  # noqa: BLE001
        _embedder = False
        return None


def create_vector_index(collection, field: str = "embedding", dim: int = EMBED_DIM):
    """Create an Atlas Vector Search index on `field` (idempotent-ish)."""
    from pymongo.operations import SearchIndexModel
    model = SearchIndexModel(
        definition={"fields": [{"type": "vector", "path": field,
                                "numDimensions": dim, "similarity": "cosine"}]},
        name="vector_index", type="vectorSearch")
    try:
        collection.create_search_index(model=model)
        return "created"
    except Exception as e:  # noqa: BLE001 — already exists / unsupported tier
        return f"skip ({type(e).__name__}: {str(e)[:80]})"


class Store:
    def __init__(self, run_id: str, out_dir: str = "runs", uri: str | None = None,
                 db: str | None = None) -> None:
        self.run_id = run_id
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self._ep = open(f"{out_dir}/episodes.jsonl", "a")
        self._db = None
        # Accept either name: MONGODB_URI (generic) or MONGODB_ATLAS_URI (Atlas-specific).
        uri = uri or os.environ.get("MONGODB_URI") or os.environ.get("MONGODB_ATLAS_URI")
        if uri:
            try:
                from pymongo import MongoClient
                c = MongoClient(uri, serverSelectionTimeoutMS=2500)
                c.admin.command("ping")
                self._db = c[db or os.environ.get("MONGODB_DB", "persona_ci")]
                print(f"[store] Mongo connected: {self._db.name}")
            except Exception as e:  # noqa: BLE001 — fall back to JSONL, never crash
                print(f"[store] Mongo unavailable ({type(e).__name__}); JSONL only")
                self._db = None

    @property
    def backend(self) -> str:
        return "mongo+jsonl" if self._db is not None else "jsonl"

    def save_personas(self, personas, path: str = "personas.json", embed: bool = True) -> None:
        docs = [asdict(p) if is_dataclass(p) else dict(p) for p in personas]
        # JSONL/local copy stays embedding-free (small); Mongo/Atlas gets the vector.
        Path(path).write_text(json.dumps(docs, ensure_ascii=False, indent=2))
        if self._db is not None:
            mdocs = []
            for d in docs:
                m = {"run_id": self.run_id, **d}
                if embed:
                    v = _embed(d.get("descriptor", "") + " " + " ".join(d.get("care_vector", {})))
                    if v:
                        m["embedding"] = v
                mdocs.append(m)
            self._db.personas.delete_many({"run_id": self.run_id})
            self._db.personas.insert_many(mdocs)

    def write_episode(self, doc: dict, embed: bool = True) -> None:
        self._ep.write(json.dumps({k: v for k, v in doc.items() if k != "embedding"}) + "\n")
        self._ep.flush()
        if self._db is not None:
            m = {"run_id": self.run_id, **doc}
            if embed and "embedding" not in m:
                v = _embed(str(doc.get("request", "")) + " " + str(doc.get("action", "")))
                if v:
                    m["embedding"] = v
            self._db.episodes.insert_one(m)

    def close(self) -> None:
        self._ep.close()
