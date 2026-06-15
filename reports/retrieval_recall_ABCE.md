# Retrieval recall diagnosis

- retrieved: `data/HOVER_subset/bit_plan7.0_graphrag/dev_2_retrieved_0_200.ABCE.json`
- raw HOVER: `/mnt/data/yangjun/data/HOVER/data/raw/hover_dev_release_v1.1.json`
- claims analysed: 200 (skipped 0 without gold)

## 1. Recall by hop

| hop | N | pool@docs | perFact_top5@docs | selected@docs | selected@sents |
|---|---:|---:|---:|---:|---:|
| 2 | 57 |  85.1% |  73.7% |  74.6% |  69.5% |
| 3 | 91 |  69.2% |  55.3% |  57.9% |  46.1% |
| 4 | 52 |  64.4% |  50.5% |  54.3% |  40.2% |
| overall | 200 |  72.5% |  59.3% |  61.7% |  51.2% |

**Reading the columns**

- `pool@docs`: gold titles present in BM25 candidate_docs pool. If this drops on deep hops, the BM25 stage itself is leaking — no downstream ranker can recover.
- `perFact_top5@docs`: gold titles covered by the union of every fact's top-5 candidates. Lower bound — the upstream pipeline only persists top-5, but the actual ranker considers the full ~25-30 candidate set per fact.
- `selected@docs` / `selected@sents`: doc- and sentence-level recall of what the per-fact LLM actually sees.

## 2. Failure-mode buckets

| hop | full_hit | pool_miss | ranking_miss | zero pool | zero selected |
|---|---:|---:|---:|---:|---:|
| 2 |  50.9% |  24.6% |  24.6% |   5.3% |   7.0% |
| 3 |   3.3% |  71.4% |  25.3% |   0.0% |   0.0% |
| 4 |   1.9% |  90.4% |   7.7% |   0.0% |   1.9% |
| overall |  16.5% |  63.0% |  20.5% |   1.5% |   2.5% |

- `pool_miss`: at least one gold title never entered the candidate pool → recall bottleneck (BM25 fix needed).
- `ranking_miss`: every gold title was in the pool, but at least one (title, sent_idx) didn't make selected → ranking / assembly bottleneck.
- `zero pool` / `zero selected`: catastrophic cases where the pool / selected set has zero overlap with gold.
