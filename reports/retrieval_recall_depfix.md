# Retrieval recall diagnosis

- retrieved: `./data/HOVER_subset/bit_plan9.0_graphrag/dev_2_retrieved_depfix_0_200.json`
- raw HOVER: `/mnt/data/yangjun/data/HOVER/data/raw/hover_dev_release_v1.1.json`
- claims analysed: 200 (skipped 0 without gold)

## 1. Recall by hop

| hop | N | pool@docs | perFact_top5@docs | selected@docs | selected@sents |
|---|---:|---:|---:|---:|---:|
| 2 | 57 |  87.7% |  75.4% |  76.3% |  66.6% |
| 3 | 91 |  81.0% |  63.4% |  64.5% |  48.9% |
| 4 | 52 |  78.4% |  59.6% |  60.6% |  46.9% |
| overall | 200 |  82.2% |  65.8% |  66.8% |  53.4% |

**Reading the columns**

- `pool@docs`: gold titles present in BM25 candidate_docs pool. If this drops on deep hops, the BM25 stage itself is leaking — no downstream ranker can recover.
- `perFact_top5@docs`: gold titles covered by the union of every fact's top-5 candidates. Lower bound — the upstream pipeline only persists top-5, but the actual ranker considers the full ~25-30 candidate set per fact.
- `selected@docs` / `selected@sents`: doc- and sentence-level recall of what the per-fact LLM actually sees.

## 2. Failure-mode buckets

| hop | full_hit | pool_miss | ranking_miss | zero pool | zero selected |
|---|---:|---:|---:|---:|---:|
| 2 |  43.9% |  21.1% |  35.1% |   3.5% |   5.3% |
| 3 |  15.4% |  44.0% |  40.7% |   0.0% |   1.1% |
| 4 |   9.6% |  55.8% |  34.6% |   0.0% |   1.9% |
| overall |  22.0% |  40.5% |  37.5% |   1.0% |   2.5% |

- `pool_miss`: at least one gold title never entered the candidate pool → recall bottleneck (BM25 fix needed).
- `ranking_miss`: every gold title was in the pool, but at least one (title, sent_idx) didn't make selected → ranking / assembly bottleneck.
- `zero pool` / `zero selected`: catastrophic cases where the pool / selected set has zero overlap with gold.
