# Retrieval recall diagnosis

- retrieved: `data/HOVER_subset/bit_plan7.0_graphrag/dev_2_retrieved_0_200.ABE.json`
- raw HOVER: `/mnt/data/yangjun/data/HOVER/data/raw/hover_dev_release_v1.1.json`
- claims analysed: 200 (skipped 0 without gold)

## 1. Recall by hop

| hop | N | pool@docs | perFact_top5@docs | selected@docs | selected@sents |
|---|---:|---:|---:|---:|---:|
| 2 | 57 |  85.1% |  71.9% |  76.3% |  67.7% |
| 3 | 91 |  68.1% |  55.7% |  57.9% |  45.4% |
| 4 | 52 |  63.9% |  49.5% |  52.9% |  38.0% |
| overall | 200 |  71.9% |  58.7% |  61.8% |  49.9% |

**Reading the columns**

- `pool@docs`: gold titles present in BM25 candidate_docs pool. If this drops on deep hops, the BM25 stage itself is leaking — no downstream ranker can recover.
- `perFact_top5@docs`: gold titles covered by the union of every fact's top-5 candidates. Lower bound — the upstream pipeline only persists top-5, but the actual ranker considers the full ~25-30 candidate set per fact.
- `selected@docs` / `selected@sents`: doc- and sentence-level recall of what the per-fact LLM actually sees.

## 2. Failure-mode buckets

| hop | full_hit | pool_miss | ranking_miss | zero pool | zero selected |
|---|---:|---:|---:|---:|---:|
| 2 |  43.9% |  24.6% |  31.6% |   5.3% |   7.0% |
| 3 |   5.5% |  73.6% |  20.9% |   0.0% |   1.1% |
| 4 |   1.9% |  88.5% |   9.6% |   0.0% |   1.9% |
| overall |  15.5% |  63.5% |  21.0% |   1.5% |   3.0% |

- `pool_miss`: at least one gold title never entered the candidate pool → recall bottleneck (BM25 fix needed).
- `ranking_miss`: every gold title was in the pool, but at least one (title, sent_idx) didn't make selected → ranking / assembly bottleneck.
- `zero pool` / `zero selected`: catastrophic cases where the pool / selected set has zero overlap with gold.
