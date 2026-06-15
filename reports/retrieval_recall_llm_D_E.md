# Retrieval recall diagnosis

- retrieved: `data/HOVER_subset/bit_plan7.0_graphrag/dev_2_retrieved_0_200.llm_D_E.json`
- raw HOVER: `/mnt/data/yangjun/data/HOVER/data/raw/hover_dev_release_v1.1.json`
- claims analysed: 200 (skipped 0 without gold)

## 1. Recall by hop

| hop | N | pool@docs | perFact_top5@docs | selected@docs | selected@sents |
|---|---:|---:|---:|---:|---:|
| 2 | 57 |  84.2% |  71.9% |  75.4% |  66.9% |
| 3 | 91 |  62.6% |  54.6% |  57.9% |  48.4% |
| 4 | 52 |  55.8% |  49.5% |  51.0% |  39.6% |
| overall | 200 |  67.0% |  58.2% |  61.1% |  51.3% |

**Reading the columns**

- `pool@docs`: gold titles present in BM25 candidate_docs pool. If this drops on deep hops, the BM25 stage itself is leaking — no downstream ranker can recover.
- `perFact_top5@docs`: gold titles covered by the union of every fact's top-5 candidates. Lower bound — the upstream pipeline only persists top-5, but the actual ranker considers the full ~25-30 candidate set per fact.
- `selected@docs` / `selected@sents`: doc- and sentence-level recall of what the per-fact LLM actually sees.

## 2. Failure-mode buckets

| hop | full_hit | pool_miss | ranking_miss | zero pool | zero selected |
|---|---:|---:|---:|---:|---:|
| 2 |  42.1% |  26.3% |  31.6% |   5.3% |   7.0% |
| 3 |   6.6% |  78.0% |  15.4% |   1.1% |   1.1% |
| 4 |   1.9% |  96.2% |   1.9% |   0.0% |   1.9% |
| overall |  15.5% |  68.0% |  16.5% |   2.0% |   3.0% |

- `pool_miss`: at least one gold title never entered the candidate pool → recall bottleneck (BM25 fix needed).
- `ranking_miss`: every gold title was in the pool, but at least one (title, sent_idx) didn't make selected → ranking / assembly bottleneck.
- `zero pool` / `zero selected`: catastrophic cases where the pool / selected set has zero overlap with gold.
