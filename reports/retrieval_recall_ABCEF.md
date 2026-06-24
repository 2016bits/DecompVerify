# Retrieval recall diagnosis

- retrieved: `data/HOVER_subset/azure_plan7.0_graphrag/dev_2_retrieved_0_200.ABCEF.json`
- raw HOVER: `/mnt/data/yangjun/data/HOVER/data/raw/hover_dev_release_v1.1.json`
- claims analysed: 200 (skipped 0 without gold)

## 1. Recall by hop

| hop | N | pool@docs | perFact_top5@docs | selected@docs | selected@sents |
|---|---:|---:|---:|---:|---:|
| 2 | 57 |  86.0% |  74.6% |  76.3% |  69.2% |
| 3 | 91 |  71.1% |  55.7% |  59.0% |  48.2% |
| 4 | 52 |  69.2% |  51.4% |  55.3% |  41.1% |
| overall | 200 |  74.8% |  60.0% |  63.0% |  52.3% |

**Reading the columns**

- `pool@docs`: gold titles present in BM25 candidate_docs pool. If this drops on deep hops, the BM25 stage itself is leaking — no downstream ranker can recover.
- `perFact_top5@docs`: gold titles covered by the union of every fact's top-5 candidates. Lower bound — the upstream pipeline only persists top-5, but the actual ranker considers the full ~25-30 candidate set per fact.
- `selected@docs` / `selected@sents`: doc- and sentence-level recall of what the per-fact LLM actually sees.

## 2. Failure-mode buckets

| hop | full_hit | pool_miss | ranking_miss | zero pool | zero selected |
|---|---:|---:|---:|---:|---:|
| 2 |  49.1% |  22.8% |  28.1% |   5.3% |   7.0% |
| 3 |   7.7% |  67.0% |  25.3% |   0.0% |   1.1% |
| 4 |   3.8% |  76.9% |  19.2% |   0.0% |   1.9% |
| overall |  18.5% |  57.0% |  24.5% |   1.5% |   3.0% |

- `pool_miss`: at least one gold title never entered the candidate pool → recall bottleneck (BM25 fix needed).
- `ranking_miss`: every gold title was in the pool, but at least one (title, sent_idx) didn't make selected → ranking / assembly bottleneck.
- `zero pool` / `zero selected`: catastrophic cases where the pool / selected set has zero overlap with gold.
