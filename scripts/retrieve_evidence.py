"""GraphRAG-style evidence retrieval for the DecompVerify pipeline.

This script sits between ``generate_question.py`` and ``get_answer.py``.
It reads the ``questions`` artefact produced by question planning,
runs a per-claim BM25 → heterogeneous graph → fact-aware PPR pipeline,
and writes a new artefact that mirrors the questions file but adds:

* ``retrieved_evidence``        – flat string usable in place of
  ``gold_evidence``.
* ``retrieved_evidence_per_fact`` – per-fact slice for finer-grained
  prompts.
* ``retrieval_meta``            – graph stats and the per-fact
  candidate breakdown for debugging.

Run, from the repo root::

    python scripts/retrieve_evidence.py \\
        --plan bit_plan7.0_graphrag --dataset HOVER_subset \\
        --data_type dev --class_num 2 --start 0 --end 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Dict, List

from tqdm import tqdm

from graph_rag.bm25_index import BM25Searcher, build_index, DEFAULT_INDEX_PATH
from graph_rag.encoder import SentenceEncoder, get_encoder
from graph_rag.retriever import RetrievalConfig, retrieve_for_claim


def _extract_initial_bindings(decomposition: dict) -> Dict[str, str]:
    bindings: Dict[str, str] = {}
    for slot, info in (decomposition.get("entity_slots", {}) or {}).items():
        if not isinstance(info, dict):
            continue
        value = info.get("value")
        if isinstance(value, str) and value.strip():
            bindings[slot] = value.strip()
    return bindings


def process_one(
    item: dict,
    *,
    searcher: BM25Searcher,
    encoder: SentenceEncoder,
    config: RetrievalConfig,
) -> dict:
    claim = item.get("claim", "")
    question_plan = item.get("question_plan") or {}
    question_items = question_plan.get("question_items", []) or []
    initial_bindings = _extract_initial_bindings(item.get("decomposition", {}) or {})

    t_start = time.time()
    try:
        result = retrieve_for_claim(
            claim=claim,
            question_items=question_items,
            searcher=searcher,
            encoder=encoder,
            config=config,
            initial_bindings=initial_bindings,
        )
    except Exception as exc:  # pragma: no cover - defensive
        result = {
            "candidate_docs": [],
            "graph_stats": {"error": str(exc)},
            "fact_results": [],
            "selected_sentences": [],
            "evidence_text": "",
            "evidence_per_fact": {},
        }
    item_elapsed = time.time() - t_start
    if item_elapsed > 10.0:
        # Surface unusually slow claims so we can spot adversarial graphs.
        print(f"[retrieve_evidence] slow claim {item.get('id')} "
              f"(hops={item.get('num_hops')}): {item_elapsed:.1f}s",
              flush=True)

    out = dict(item)
    out["retrieved_evidence"] = result.get("evidence_text", "")
    out["retrieved_evidence_per_fact"] = result.get("evidence_per_fact", {})
    out["retrieval_meta"] = {
        "candidate_docs": result.get("candidate_docs", []),
        "graph_stats": result.get("graph_stats", {}),
        "selected_sentences": [
            {
                "sid": s["sid"],
                "title": s["title"],
                "sent_idx": s["sent_idx"],
                "fact_id": s.get("fact_id"),
                "composite_score": s.get("composite_score"),
                "ce_score": s.get("ce_score"),
                "direct_support": s.get("direct_support"),
            }
            for s in result.get("selected_sentences", [])
        ],
        "assembly_summary": result.get("assembly_summary", {}),
        "fact_breakdown": [
            {
                "fact_id": fr["fact_id"],
                "rely_on": fr.get("rely_on", []),
                "critical": fr.get("critical", False),
                "seed_summary": fr.get("seed_summary", {}),
                "num_candidates": len(fr.get("candidates", [])),
                "num_direct": len(fr.get("direct_supports", [])),
                "num_bridge": len(fr.get("bridge_supports", [])),
                "top_candidates": [
                    {
                        "sid": c["sid"],
                        "title": c["title"],
                        "sent_idx": c["sent_idx"],
                        "composite_score": c["composite_score"],
                        "ce_score": c["ce_score"],
                        "direct_support": c["direct_support"],
                    }
                    for c in fr.get("candidates", [])[:5]
                ],
            }
            for fr in result.get("fact_results", [])
        ],
    }
    return out


def _resolve_path(template: str, args) -> str:
    return (
        template
        .replace("[DATA]", args.dataset)
        .replace("[TYPE]", args.data_type)
        .replace("[CLASS]", args.class_num)
        .replace("[T]", args.t)
        .replace("[S]", str(args.start))
        .replace("[E]", str(args.end))
        .replace("[PLAN]", args.plan)
    )


def main(args):
    in_path = _resolve_path(args.in_path, args)
    out_path = _resolve_path(args.out_path, args)

    with open(in_path, "r", encoding="utf-8") as fh:
        raws = json.load(fh)
    raws = raws[args.start:args.end]

    # Index bootstrap – build on demand if missing.
    if not os.path.exists(args.bm25_index):
        print(f"[retrieve_evidence] BM25 index not found at {args.bm25_index}; "
              f"building from {args.corpus} (this can take ~10 min).")
        build_index(args.corpus, args.bm25_index)

    searcher = BM25Searcher(args.bm25_index)
    encoder = get_encoder(args.encoder_path)

    config = RetrievalConfig(
        k_claim=args.k_claim,
        k_fact=args.k_fact,
        k_critical=args.k_critical,
        k_constraint=args.k_constraint,
        max_docs=args.max_docs,
        sem_top_k=args.sem_top_k,
        sem_threshold=args.sem_threshold,
        fact_top_k_candidates=args.fact_top_k_candidates,
        fact_top_k_final=args.fact_top_k_final,
        final_max_sentences=args.final_max_sentences,
        final_max_docs=args.final_max_docs,
    )

    fn = partial(process_one, searcher=searcher, encoder=encoder, config=config)

    results: List[dict] = []
    start_time = time.time()
    log_every = max(1, args.log_every)
    if args.max_workers <= 1:
        for i, item in enumerate(raws):
            t0 = time.time()
            results.append(fn(item))
            if (i + 1) % log_every == 0 or (i + 1) == len(raws):
                avg = (time.time() - start_time) / (i + 1)
                print(f"[retrieve_evidence] {i+1}/{len(raws)}  "
                      f"last={time.time()-t0:.2f}s  avg={avg:.2f}s/claim",
                      flush=True)
    else:
        # The encoder uses a single GPU; we keep concurrency low. BM25 and
        # graph work are CPU-bound and threaded fine under the GIL since
        # most heavy ops release it (NumPy, networkx C extensions).
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(fn, item) for item in raws]
            for i, fut in enumerate(as_completed(futures)):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    print(f"[retrieve_evidence] error on item: {exc}",
                          flush=True)
                if (i + 1) % log_every == 0 or (i + 1) == len(futures):
                    avg = (time.time() - start_time) / (i + 1)
                    print(f"[retrieve_evidence] {i+1}/{len(futures)}  "
                          f"avg={avg:.2f}s/claim", flush=True)

    results.sort(key=lambda d: d.get("id", ""))

    elapsed = time.time() - start_time
    print(f"[retrieve_evidence] processed {len(results)} claims in "
          f"{elapsed:.1f}s ({elapsed/max(len(results),1):.2f}s/claim).",
          flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"[retrieve_evidence] wrote {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="HOVER_subset")
    parser.add_argument("--data_type", type=str, default="dev")
    parser.add_argument("--class_num", type=str, default="2")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument("--plan", type=str, default="bit_plan7.0")
    parser.add_argument("--t", type=str, default="")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--in_path", type=str,
                        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_questions_[T][S]_[E].json")
    parser.add_argument("--out_path", type=str,
                        default="./data/[DATA]/[PLAN]/[TYPE]_[CLASS]_retrieved_[T][S]_[E].json")

    parser.add_argument("--bm25_index", type=str, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--corpus", type=str,
                        default="/mnt/data/yangjun/data/HOVER/corpus/jsonl_corpus/hotpotqa_corpus.jsonl")
    parser.add_argument("--encoder_path", type=str,
                        default="/data/public/LLM/all-MiniLM-L6-v2")

    # role-aware BM25 quotas
    parser.add_argument("--k_claim", type=int, default=8)
    parser.add_argument("--k_fact", type=int, default=6)
    parser.add_argument("--k_critical", type=int, default=8)
    parser.add_argument("--k_constraint", type=int, default=4)
    parser.add_argument("--max_docs", type=int, default=30)

    # graph and PPR
    parser.add_argument("--sem_top_k", type=int, default=5)
    parser.add_argument("--sem_threshold", type=float, default=0.55)
    parser.add_argument("--fact_top_k_candidates", type=int, default=25)
    parser.add_argument("--fact_top_k_final", type=int, default=5)

    # assembly
    parser.add_argument("--final_max_sentences", type=int, default=8)
    parser.add_argument("--final_max_docs", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10,
                        help="Print a progress line every N processed claims.")
    main(parser.parse_args())
