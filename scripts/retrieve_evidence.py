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
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

from tqdm import tqdm

from graph_rag.bm25_index import BM25Searcher, build_index, DEFAULT_INDEX_PATH
from graph_rag.encoder import SentenceEncoder, get_encoder
from graph_rag.reranker import CrossEncoderReranker, get_reranker
from graph_rag.retriever import (
    RetrievalConfig,
    _is_placeholder_binding,
    retrieve_for_claim,
)
from dependency_repair import VAR_PATTERN, repair_item_dependencies


# --------------------------------------------------------------------------- LLM-assisted binding
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _build_llm_client(plan: str):
    """Build an OpenAI-compatible client + model name from ``plan`` prefix.

    Mirrors the logic in ``scripts/get_answer.py:build_client_and_model`` but
    only returns the LLM call surface needed for binding verification (no
    system prompt, no extra_body). Honors the ``LLM_FALLBACK=aiping`` env
    override so the same code works on/off BIT VPN.
    """
    from openai import OpenAI

    plan = (plan or "").lower()
    fallback = os.getenv("LLM_FALLBACK", "").lower() == "aiping"

    if plan.startswith("azure"):
        return OpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            base_url="https://open1027.openai.azure.com/openai/v1/",
        ), "gpt-4o"
    if plan.startswith("qc_plan") or fallback or plan.startswith("qwen_plan"):
        return OpenAI(
            api_key=os.environ["AIPING_API_KEY"],
            base_url="https://www.aiping.cn/api/v1",
        ), "DeepSeek-V3.2"
    if plan.startswith("bit_plan"):
        return OpenAI(
            api_key=os.environ["BIT_API_KEY"],
            base_url="https://maas.bit.edu.cn/v1",
        ), "DeepSeek-V3.2"
    # default: try BIT
    return OpenAI(
        api_key=os.environ["BIT_API_KEY"],
        base_url="https://maas.bit.edu.cn/v1",
    ), "DeepSeek-V3.2"


def _format_constraint(qitem: dict) -> str:
    cons = qitem.get("constraint") or {}
    bits: List[str] = []
    for slot_name in ("time", "exact_name", "location", "title_role",
                       "unique_modifier", "quantity", "comparison"):
        for v in cons.get(slot_name, []) or []:
            if isinstance(v, str) and v.strip() and not v.startswith("?"):
                bits.append(f"{slot_name}={v}")
    if cons.get("negation"):
        bits.append(f"negation={cons['negation']}")
    return "; ".join(bits) if bits else "(none)"


def _format_fact_with_bindings(fact_text: str, bindings: Dict[str, str]) -> str:
    """Substitute resolved (non-placeholder) bindings into the fact text."""
    out = fact_text or ""
    for var, val in bindings.items():
        if not val or _is_placeholder_binding(val):
            continue
        out = out.replace(var, val)
    return out


def make_llm_binding_fn(plan: str, max_retries: int = 2, debug: bool = False):
    """Return a closure ``(qitem, top_candidates, bindings) -> Optional[(slot, value)]``.

    The closure asks the LLM to identify which named entity the fact's
    ``answer_slot`` refers to, given the top retrieval candidates. It
    returns ``None`` when the LLM is not confident, so the caller can
    refuse to write a noisy binding.
    """
    client, model = _build_llm_client(plan)

    def _fn(qitem: dict, top_cands: List[dict],
            bindings: Dict[str, str]) -> Optional[Tuple[str, str]]:
        slot = qitem.get("answer_slot")
        if not isinstance(slot, str) or not slot.startswith("?"):
            return None
        if not top_cands:
            return None

        fact_text = _format_fact_with_bindings(
            qitem.get("fact_text", ""), bindings)
        constraint_str = _format_constraint(qitem)
        cand_lines = []
        for i, c in enumerate(top_cands[:5], 1):
            title = c.get("title", "")
            text = (c.get("text") or "").strip().replace("\n", " ")
            if len(text) > 350:
                text = text[:350] + "..."
            cand_lines.append(f"[{i}] (from \"{title}\") {text}")
        cand_block = "\n".join(cand_lines)

        prompt = (
            "You are resolving a variable in an atomic fact during fact verification.\n"
            "Pick the named entity that the variable should bind to, based on the "
            "evidence sentences below.\n\n"
            f"Fact statement: {fact_text}\n"
            f"Constraints on the answer: {constraint_str}\n"
            f"Variable to bind: {slot}\n\n"
            "Evidence sentences (top retrieval candidates):\n"
            f"{cand_block}\n\n"
            "Rules:\n"
            f"- \"value\" must be the bare Wikipedia-style entity name for {slot} "
            "(e.g. \"Karl Kraepelin\", \"Karachi Blues\", \"Iqbal Stadium\").\n"
            "- Do NOT use descriptions like \"a German naturalist\" — give the actual name.\n"
            "- If no sentence above unambiguously supports a value, return value=null.\n"
            "- \"evidence_idx\" is the [N] index supporting your pick (0 if value is null).\n\n"
            "Respond with ONLY a JSON object on one line, no markdown fences, no commentary:\n"
            "{\"value\": \"<entity name or null>\", \"evidence_idx\": <integer>}"
        )

        last_err = None
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=128,
                )
                content = (resp.choices[0].message.content or "").strip()
                content = _JSON_FENCE_RE.sub("", content).strip()
                # find first { ... } if there's noise
                m = re.search(r"\{[^{}]*\}", content)
                if m:
                    content = m.group(0)
                parsed = json.loads(content)
                value = parsed.get("value")
                if not isinstance(value, str):
                    return None
                value = value.strip()
                if not value or value.lower() in {"null", "none", "n/a", "nil"}:
                    if debug:
                        print(f"[llm_binding] {qitem.get('fact_id')} {slot}: null",
                              flush=True)
                    return None
                if debug:
                    print(f"[llm_binding] {qitem.get('fact_id')} {slot} = {value}",
                          flush=True)
                return (slot, value)
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                if debug:
                    print(f"[llm_binding] error {qitem.get('fact_id')}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                return None
        return None

    return _fn


def make_leaf_cand_fn(plan: str, max_candidates: int = 3,
                       max_retries: int = 2, debug: bool = False):
    """Return ``(claim, qitem, bindings) -> List[str]`` for Option F1.

    Asks an LLM (gpt-4o on ``azure*`` plans, DeepSeek-V3.2 elsewhere) to
    generate up to ``max_candidates`` wiki-style title candidates for
    the fact's answer slot. The retriever immediately title-only BM25s
    each candidate, so even imperfect names are useful — we win as long
    as one of the candidates is the actual gold title or a near-match.

    Failure modes (LLM error, JSON parse fail, empty answer) return ``[]``.
    """
    client, model = _build_llm_client(plan)

    def _format_bindings(b: Dict[str, str]) -> str:
        if not b:
            return "(none)"
        bits = []
        for k, v in b.items():
            if isinstance(v, str) and v and not _is_placeholder_binding(v):
                bits.append(f"{k} = {v}")
        return "; ".join(bits) if bits else "(none)"

    def _fn(claim: str, qitem: dict, bindings: Dict[str, str]) -> List[str]:
        fact_text = qitem.get("fact_text", "") or ""
        slot = qitem.get("answer_slot") or ""
        bind_str = _format_bindings(bindings)
        cons_str = _format_constraint(qitem)

        prompt = (
            "You are helping a fact-checking system locate the right Wikipedia "
            "pages for unresolved entities in a multi-hop claim. For the "
            "sub-fact below, propose Wikipedia-style entity titles that the "
            "answer is most likely to be.\n\n"
            f"Claim: {claim}\n\n"
            f"Sub-fact: {fact_text}\n"
            f"Variable to resolve: {slot or '(unspecified)'}\n"
            f"Constraints on the answer: {cons_str}\n"
            f"Resolved bindings from parent sub-facts: {bind_str}\n\n"
            "Rules:\n"
            "- Each candidate must be a named entity / proper noun.\n"
            "- Give the bare Wikipedia title (e.g. \"Gaddafi Stadium\", "
            "\"Scorpion\", \"Karl Kraepelin\"), NOT a description.\n"
            "- Prefer specific over generic (\"Scorpion\" beats \"Arachnid\").\n"
            "- If uncertain between near-synonyms, list both.\n"
            f"- Give at most {max_candidates} candidates; you may give fewer.\n\n"
            "Respond with ONLY a JSON object on one line, no fences:\n"
            '{"candidates": ["TitleA", "TitleB", "TitleC"]}'
        )

        last_err = None
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=160,
                )
                content = (resp.choices[0].message.content or "").strip()
                content = _JSON_FENCE_RE.sub("", content).strip()
                m = re.search(r"\{[^{}]*\}", content)
                if m:
                    content = m.group(0)
                parsed = json.loads(content)
                cands = parsed.get("candidates") or []
                out: List[str] = []
                seen: set = set()
                for c in cands:
                    if not isinstance(c, str):
                        continue
                    c = c.strip().strip(".,;:")
                    if not c or len(c) < 3:
                        continue
                    key = c.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(c)
                    if len(out) >= max_candidates:
                        break
                if debug:
                    print(f"[leaf_cand] {qitem.get('fact_id')} -> {out}",
                          flush=True)
                return out
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                if debug:
                    print(f"[leaf_cand] error {qitem.get('fact_id')}: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                return []
        return []

    return _fn


def _extract_initial_bindings(decomposition: dict) -> Dict[str, str]:
    bindings: Dict[str, str] = {}
    for slot, info in (decomposition.get("entity_slots", {}) or {}).items():
        if not isinstance(info, dict):
            continue
        value = info.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        # Descriptions such as "a Catalan football club" should remain
        # unresolved so retrieval can bind them from evidence instead of
        # substituting prose into downstream queries.
        if VAR_PATTERN.search(value) or _is_placeholder_binding(value):
            continue
        bindings[slot] = value
    return bindings


def process_one(
    item: dict,
    *,
    searcher: BM25Searcher,
    encoder: SentenceEncoder,
    config: RetrievalConfig,
    llm_binding_fn: Optional[Callable] = None,
    reranker: Optional[CrossEncoderReranker] = None,
    leaf_cand_fn: Optional[Callable] = None,
    dependency_repair: bool = False,
) -> dict:
    repair_summary = None
    if dependency_repair:
        item, repair_summary = repair_item_dependencies(item)

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
            llm_binding_fn=llm_binding_fn,
            num_hops=item.get("num_hops"),
            reranker=reranker,
            leaf_cand_fn=leaf_cand_fn,
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
    if repair_summary is not None:
        out["retrieval_meta"]["dependency_repair"] = repair_summary
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
        llm_binding_all_facts=bool(args.llm_binding_all_facts),
        llm_binding_no_heuristic_fallback=bool(
            args.llm_binding_no_heuristic_fallback),
        llm_binding_top_k=args.llm_binding_top_k,
        min_hops_for_binding=args.min_hops_for_binding,
        require_rely_on_for_binding=not args.no_require_rely_on,
        # Option B
        enable_title_anchor=not args.no_title_anchor,
        k_title_anchor=args.k_title_anchor,
        title_anchor_boost=args.title_anchor_boost,
        # Option E
        chain_depth_for_boost=args.chain_depth_for_boost,
        chain_boost_factor=args.chain_boost_factor,
        max_docs_per_extra_hop=args.max_docs_per_extra_hop,
        # Option A
        enable_soft_expand=not args.no_soft_expand,
        soft_expand_k_query=args.soft_expand_k_query,
        soft_expand_max_new=args.soft_expand_max_new,
        # Option C
        reranker_blend=args.reranker_blend,
        reranker_max_candidates=args.reranker_max_candidates,
        direct_support_offset=args.direct_support_offset,
        # Option F1
        leaf_cand_min_depth=args.leaf_cand_min_depth,
        leaf_cand_max_new_docs=args.leaf_cand_max_new_docs,
        unresolved_binding_override=not args.no_unresolved_binding_override,
        unresolved_leaf_cand=not args.no_unresolved_leaf_cand,
    )

    # Option C: load the cross-encoder reranker (lazy — no-op if disabled).
    reranker = None
    if args.reranker_path:
        print(f"[retrieve_evidence] loading reranker {args.reranker_path} "
              f"(device={args.reranker_device or 'auto'}, "
              f"batch={args.reranker_batch_size}, fp16={not args.reranker_fp32})",
              flush=True)
        reranker = get_reranker(
            model_path=args.reranker_path,
            device=args.reranker_device or None,
            batch_size=args.reranker_batch_size,
            max_length=args.reranker_max_length,
            use_fp16=not args.reranker_fp32,
        )
        if reranker is None:
            print("[retrieve_evidence] reranker load failed — falling back to "
                  "heuristic-only scoring", flush=True)

    llm_binding_fn = None
    if args.use_llm_binding:
        print(f"[retrieve_evidence] enabling LLM-assisted binding "
              f"(all_facts={config.llm_binding_all_facts}, "
              f"top_k={config.llm_binding_top_k})", flush=True)
        llm_binding_fn = make_llm_binding_fn(
            plan=args.plan, debug=bool(args.llm_binding_debug))

    leaf_cand_fn = None
    if args.use_leaf_cand:
        print(f"[retrieve_evidence] enabling Option F1 leaf-candidate gen "
              f"(deepest_leaf_only=True, min_depth={args.leaf_cand_min_depth}, "
              f"max_candidates={args.leaf_cand_max_candidates}, "
              f"max_new_docs={args.leaf_cand_max_new_docs})", flush=True)
        leaf_cand_fn = make_leaf_cand_fn(
            plan=args.plan,
            max_candidates=args.leaf_cand_max_candidates,
            debug=bool(args.leaf_cand_debug),
        )

    fn = partial(process_one, searcher=searcher, encoder=encoder, config=config,
                 llm_binding_fn=llm_binding_fn, reranker=reranker,
                 leaf_cand_fn=leaf_cand_fn,
                 dependency_repair=bool(args.dependency_repair))

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

    # LLM-assisted binding (Option C)
    parser.add_argument("--use_llm_binding", action="store_true",
                        help="Use an LLM to verify/select the answer-slot "
                             "binding instead of the heuristic.")
    parser.add_argument("--llm_binding_all_facts", action="store_true",
                        help="Call the LLM on every fact (default: critical only).")
    parser.add_argument("--llm_binding_no_heuristic_fallback",
                        action="store_true", default=True,
                        help="Skip the heuristic fallback when the LLM returns "
                             "null (default: True).")
    parser.add_argument("--llm_binding_top_k", type=int, default=5,
                        help="Number of top candidate sentences passed to the LLM.")
    parser.add_argument("--llm_binding_debug", action="store_true",
                        help="Verbose logging of every LLM binding decision.")

    # Options D + E (binding): gate binding by claim/fact structure
    parser.add_argument("--min_hops_for_binding", type=int, default=3,
                        help="(Option D) Disable binding propagation for "
                             "claims with num_hops below this. Set to 0 to "
                             "disable the gate. Default 3.")
    parser.add_argument("--no_require_rely_on", action="store_true",
                        help="(Inverse of Option E binding) Allow binding on "
                             "root facts that have no rely_on. Default off — "
                             "root facts skip binding.")

    # Option B: title-only anchor channel
    parser.add_argument("--no_title_anchor", action="store_true",
                        help="(Option B) Disable the title-only BM25 channel.")
    parser.add_argument("--k_title_anchor", type=int, default=3,
                        help="(Option B) top-k for each title-only query.")
    parser.add_argument("--title_anchor_boost", type=float, default=1.5,
                        help="(Option B) score multiplier for title-anchor hits.")

    # Option E (recall): dynamic budgets
    parser.add_argument("--chain_depth_for_boost", type=int, default=2,
                        help="(Option E) facts with rely_on chain depth >= "
                             "this value get k_fact * chain_boost_factor.")
    parser.add_argument("--chain_boost_factor", type=float, default=1.5,
                        help="(Option E) k_fact / k_critical multiplier for "
                             "deep-chain facts.")
    parser.add_argument("--max_docs_per_extra_hop", type=int, default=10,
                        help="(Option E) extra max_docs budget per num_hops "
                             "above 2. Set 0 to disable.")

    # Option A: forced soft expansion
    parser.add_argument("--no_soft_expand", action="store_true",
                        help="(Option A) Disable the parent-augmented BM25 "
                             "pre-pass that runs before per-fact retrieval.")
    parser.add_argument("--soft_expand_k_query", type=int, default=6,
                        help="(Option A) top-k for the augmented BM25 query.")
    parser.add_argument("--soft_expand_max_new", type=int, default=5,
                        help="(Option A) max new docs added per soft expand.")

    # Option C: cross-encoder reranker
    parser.add_argument("--reranker_path", type=str, default="",
                        help="(Option C) path to a cross-encoder model. Pass "
                             "/mnt/data/hezhisheng/models/bge-reranker-v2-m3 "
                             "to enable. Empty string disables reranking.")
    parser.add_argument("--reranker_device", type=str, default="",
                        help="(Option C) cuda / cpu. Empty = auto-detect.")
    parser.add_argument("--reranker_batch_size", type=int, default=64,
                        help="(Option C) reranker batch size per fact.")
    parser.add_argument("--reranker_max_length", type=int, default=256,
                        help="(Option C) max token length per (q,p) pair.")
    parser.add_argument("--reranker_fp32", action="store_true",
                        help="(Option C) disable fp16 (CPU or debugging).")
    parser.add_argument("--reranker_blend", type=float, default=0.5,
                        help="(Option C) weight of reranker score in ce_score. "
                             "0.0 = ignore reranker, 1.0 = pure reranker.")
    parser.add_argument("--reranker_max_candidates", type=int, default=60,
                        help="(Option C) max candidates per fact sent to GPU.")
    parser.add_argument("--direct_support_offset", type=float, default=0.0,
                        help="(Option C.1) Constant added to every ce_score "
                             "threshold in _direct_support_pass. Use ~0.10 "
                             "when reranker is enabled to match its higher "
                             "ce_score mean. 0.0 = legacy gate.")

    # Option F1: LLM leaf-candidate answer generation
    parser.add_argument("--use_leaf_cand", action="store_true",
                        help="(Option F1) For each deepest leaf fact whose "
                             "chain depth is >= leaf_cand_min_depth, ask the "
                             "LLM to propose wiki title candidates, then "
                             "title-only BM25 each. Routes through the same "
                             "plan client.")
    parser.add_argument("--leaf_cand_min_depth", type=int, default=2,
                        help="(Option F1) minimum chain depth for firing on "
                             "a deepest leaf fact. 2 skips shallow chains.")
    parser.add_argument("--leaf_cand_max_candidates", type=int, default=3,
                        help="(Option F1) max candidates the LLM may return.")
    parser.add_argument("--leaf_cand_max_new_docs", type=int, default=8,
                        help="(Option F1) cap on new docs added per leaf.")
    parser.add_argument("--leaf_cand_debug", action="store_true",
                        help="(Option F1) verbose logging.")

    # First-batch long-hop experiment: deterministic dependency repair plus
    # unresolved-variable retrieval triggers.
    parser.add_argument("--dependency_repair", action="store_true",
                        help="Repair missing rely_on edges by linking facts "
                             "that consume a variable to the fact whose "
                             "answer_slot produces it.")
    parser.add_argument("--no_unresolved_binding_override", action="store_true",
                        help="Disable root binding for unresolved answer slots "
                             "that are consumed downstream.")
    parser.add_argument("--no_unresolved_leaf_cand", action="store_true",
                        help="Disable leaf-candidate generation for facts with "
                             "unresolved variables.")

    main(parser.parse_args())
