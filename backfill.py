"""Backfill the D2 corpus from local JSONL into MongoDB Atlas.

Use AFTER a training run when Atlas was offline (e.g. the env-var mismatch fixed in
store.py): every episode is already in `runs/episodes.jsonl`, so we re-read it,
recompute embeddings (JSONL stores none), upsert into Atlas, and (re)create the
Atlas Vector Search index — giving the demo a fully populated, vector-searchable
corpus identical to what live mirroring would have written.

  MONGODB_ATLAS_URI=...  python backfill.py                       # runs/episodes.jsonl -> Atlas
  python backfill.py runs/episodes.jsonl --personas personas.json # also backfill personas
  python backfill.py --dry-run                                    # count + validate, NO writes, no DB needed

Idempotent: backfilled docs are tagged {"_source": "backfill"} and any previous
backfill for the same --run-id is replaced (live-written docs are never touched).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from store import _embed, create_vector_index, EMBED_DIM


def _read_jsonl(path: str) -> list[dict]:
    docs = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] skipping malformed line {ln}: {e}")
    return docs


def _connect(uri: str | None, db_name: str):
    uri = uri or os.environ.get("MONGODB_URI") or os.environ.get("MONGODB_ATLAS_URI")
    if not uri:
        sys.exit("[backfill] no MONGODB_URI / MONGODB_ATLAS_URI set — nothing to write to. "
                 "(use --dry-run to validate without a DB)")
    from pymongo import MongoClient
    c = MongoClient(uri, serverSelectionTimeoutMS=5000)
    c.admin.command("ping")
    db = c[db_name]
    print(f"[backfill] Atlas connected: db={db.name}")
    return db


def _embed_episode(d: dict):
    return _embed(f"{d.get('request', '')} {d.get('action', '')}")


def _embed_persona(d: dict):
    return _embed(f"{d.get('descriptor', '')} " + " ".join(d.get("care_vector", {})))


def _backfill_collection(coll, docs, run_id, embed_fn, do_embed):
    out = []
    n_emb = 0
    for d in docs:
        m = {"run_id": run_id, "_source": "backfill", **{k: v for k, v in d.items() if k != "embedding"}}
        if do_embed:
            v = embed_fn(d)
            if v:
                m["embedding"] = v
                n_emb += 1
        out.append(m)
    coll.delete_many({"run_id": run_id, "_source": "backfill"})
    if out:
        coll.insert_many(out)
    return len(out), n_emb


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill JSONL corpus into MongoDB Atlas.")
    ap.add_argument("episodes", nargs="?", default="runs/episodes.jsonl",
                    help="episodes JSONL path (default: runs/episodes.jsonl)")
    ap.add_argument("--personas", default=None, help="personas.json to also backfill")
    ap.add_argument("--run-id", default="grpo-ci", help="run_id tag for backfilled docs")
    ap.add_argument("--db", default=os.environ.get("MONGODB_DB", "persona_ci"))
    ap.add_argument("--uri", default=None, help="override Mongo/Atlas URI")
    ap.add_argument("--no-embed", action="store_true", help="skip embeddings (no Vector Search)")
    ap.add_argument("--no-index", action="store_true", help="skip Vector Search index creation")
    ap.add_argument("--dry-run", action="store_true", help="count + validate only; no DB, no writes")
    args = ap.parse_args()

    if not Path(args.episodes).exists():
        sys.exit(f"[backfill] episodes file not found: {args.episodes}")

    eps = _read_jsonl(args.episodes)
    outcomes = Counter(d.get("outcome", "?") for d in eps)
    print(f"[backfill] episodes: {len(eps)}  outcomes: {dict(outcomes)}")

    pers = []
    if args.personas:
        if not Path(args.personas).exists():
            sys.exit(f"[backfill] personas file not found: {args.personas}")
        pers = json.loads(Path(args.personas).read_text())
        print(f"[backfill] personas: {len(pers)}")

    do_embed = not args.no_embed

    if args.dry_run:
        probe = _embed("connectivity probe") if do_embed else None
        emb_state = (f"available ({EMBED_DIM}-d)" if probe else
                     "UNAVAILABLE (install sentence-transformers)" if do_embed else "disabled (--no-embed)")
        print(f"[backfill] DRY RUN — would write {len(eps)} episodes"
              + (f" + {len(pers)} personas" if pers else "")
              + f" to db='{args.db}' run_id='{args.run_id}'")
        print(f"[backfill] embeddings: {emb_state}")
        print("[backfill] no DB connection made, no documents written.")
        return

    if do_embed and _embed("probe") is None:
        print("[backfill] WARN: sentence-transformers unavailable — writing WITHOUT embeddings "
              "(Vector Search will be empty). Install it or pass --no-embed to silence.")
        do_embed = False

    db = _connect(args.uri, args.db)

    n, ne = _backfill_collection(db.episodes, eps, args.run_id, _embed_episode, do_embed)
    print(f"[backfill] episodes -> Atlas: {n} written, {ne} with embeddings")
    if pers:
        pn, pne = _backfill_collection(db.personas, pers, args.run_id, _embed_persona, do_embed)
        print(f"[backfill] personas -> Atlas: {pn} written, {pne} with embeddings")

    if do_embed and not args.no_index:
        print(f"[backfill] episodes vector index: {create_vector_index(db.episodes)}")
        if pers:
            print(f"[backfill] personas vector index: {create_vector_index(db.personas)}")

    print("[backfill] done.")


if __name__ == "__main__":
    main()
