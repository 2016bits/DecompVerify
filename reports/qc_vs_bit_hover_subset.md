# qc_plan7.0 vs bit_plan7.0 — HOVER_subset Comparison

Both plans nominally use **DeepSeek-V3.2**, just routed through different vendors:

| plan | endpoint | env var |
|---|---|---|
| `qc_plan7.0` | `https://www.aiping.cn/api/v1` | `AIPING_API_KEY` |
| `bit_plan7.0` | `https://maas.bit.edu.cn/v1` | `BIT_API_KEY` |

Both runs use the same prompts, the same `max_tokens=…`, `temperature=0.7`, and the same 200 HOVER_subset dev claims. Verify + aggregate are deterministic (no LLM).

---

## 1. Main pipeline numbers

| metric | qc_plan7.0 (aiping) | bit_plan7.0 (BIT MaaS) | Δ |
|---|---:|---:|---:|
| overall accuracy | **0.785** | **0.760** | −0.025 |
| macro F1 | 0.785 | 0.760 | −0.025 |
| supports F1 | 0.7839 | 0.7624 | −0.022 |
| refutes F1 | 0.7861 | 0.7576 | −0.029 |
| supports recall | 0.78 | 0.77 | −0.01 |
| refutes recall | 0.79 | 0.75 | −0.04 |
| pred dist (S/R) | 99 / 101 | 102 / 98 | — |

Per-hop:

| hop | qc acc | bit acc | Δ |
|---:|---:|---:|---:|
| 2 (N=57) | 0.807 | 0.789 | −0.018 |
| 3 (N=91) | 0.780 | 0.714 | **−0.066** |
| 4 (N=52) | 0.769 | **0.808** | **+0.039** |

The macro gap of −2.5pp hides a non-uniform pattern: hop-3 is much worse on bit, but hop-4 is actually **better** on bit.

---

## 2. Claim-level agreement (where the gap actually lives)

| | count | share |
|---|---:|---:|
| same prediction (binary) | 173 / 200 | 86.5% |
| both correct | 141 | 70.5% |
| qc only correct | 16 | 8.0% |
| bit only correct | 11 | 5.5% |
| both wrong | 32 | 16.0% |

The two providers **agree on 86.5% of claims**. The whole 2.5pp gap = `qc-only correct − bit-only correct = 16 − 11 = +5` claims net. Out of 27 disagreement cases, qc happens to be right on 5 more of them than bit.

Disagreement matrix (qc → bit):

|  | bit: S | bit: R |
|---|---:|---:|
| qc: S | 87 | 12 |
| qc: R | 15 | 86 |

3-class final label distribution (before NEI is merged into refutes):

| label | qc | bit |
|---:|---:|---:|
| supports | 99 | 102 |
| refutes | 90 | 81 |
| not enough information | 11 | 17 |

bit produces noticeably more "NEI" verdicts (17 vs 11) — i.e., it more often refuses to commit. Those NEI claims still get binned into refutes for scoring, which doesn't always match the gold.

---

## 3. Where in the pipeline does divergence appear

The pipeline is `decompose → question → answer → verify(rules) → aggregate(rules)`. Verify + aggregate are deterministic Python — so any divergence in their output must come from upstream LLM outputs.

**Stage 1 — decomposition (LLM):**

| | qc | bit |
|---|---:|---:|
| total atomic facts | 998 | 962 |
| facts per claim, mean | 4.99 | 4.81 |
| facts per claim, median | 5 | 5 |
| claims with same fact count as the other plan | 92 / 200 (46%) |

**54% of claims already get different numbers of facts at step 1.** The two providers don't even agree on the decomposition granularity for the same claim.

**Stage 4 — verify-label distribution (deterministic, but consumes LLM Q/A):**

| label | qc | bit |
|---|---:|---:|
| support | 806 (80.8%) | 790 (82.1%) |
| contradict | 131 (13.1%) | 117 (12.2%) |
| insufficient | 61 (6.1%) | 55 (5.7%) |

Aggregate ratios look similar, but the absolute fact counts differ because the decomposed sets differ.

So: **the divergence is born in the decomposition step**, then propagates (with some downstream smoothing) into the final label.

---

## 4. Ablation suite — qc vs bit, side by side

`Δ` = variant macro_F1 − full macro_F1, for the same plan.

| variant | qc F1 | qc Δ | bit F1 | bit Δ | sign agree? |
|---|---:|---:|---:|---:|:--:|
| **full** | 0.7850 | — | 0.7600 | — | — |
| random_critical | 0.7850 | 0.0000 | 0.7600 | 0.0000 | ✓ |
| no_critical_gate | 0.7850 | 0.0000 | 0.7600 | 0.0000 | ✓ |
| wo_rely_on_aggregate_only | 0.7850 | 0.0000 | 0.7600 | 0.0000 | ✓ |
| decomp_no_aggregation | 0.7850 | 0.0000 | 0.7600 | 0.0000 | ✓ |
| wo_runtime_binding | 0.7800 | −0.005 | 0.7600 | 0.0000 | ✓ |
| structural_critical | 0.7750 | −0.010 | 0.7650 | +0.005 | ✗ |
| wo_constraint_full | 0.7750 | −0.010 | 0.7749 | **+0.015** | ✗ |
| wo_adjudication | 0.7588 | −0.026 | 0.7396 | −0.020 | ✓ |
| flip_critical | 0.7596 | −0.025 | 0.7636 | +0.004 | ✗ |
| wo_value_checks | 0.7436 | −0.041 | 0.7445 | −0.016 | ✓ (mag differs) |
| wo_polarity | 0.7440 | −0.041 | 0.7298 | −0.030 | ✓ |
| majority_vote | 0.5054 | −0.280 | 0.5096 | −0.250 | ✓ |

**Observations**

- **The strongest effects are robust across both providers**: `majority_vote` collapses both runs by ~25–28pp; `wo_polarity`, `wo_value_checks`, `wo_adjudication` are all clear negatives on both sides.
- **Several zero-effect variants are identical** (`random_critical`, `no_critical_gate`, `wo_rely_on_aggregate_only`, `decomp_no_aggregation`) — these mechanisms don't fire at all on this baseline path under either provider, ruling out provider-side flukes.
- **Three variants flip sign**: `structural_critical`, `wo_constraint_full`, `flip_critical`. These are all ≤ 1.5pp moves in either direction — i.e., **inside the noise floor**. On qc they happen to land on the negative side of zero, on bit they happen to land on the positive side. With 200 samples it's coin-flip-level which way they go.

---

## 5. So why does the "same model" give different numbers?

The 2.5pp gap is **not statistically significant** at N=200. Sketch:

> Standard error of accuracy at p≈0.77 on N=200: `√(p(1-p)/n) = √(0.77·0.23/200) ≈ 0.030`.
> Observed gap = **0.025 ≈ 0.84 SE.** A two-proportion z-test on (157/200) vs (152/200) gives **p ≈ 0.57**.

So a sane prior is: the gap could easily come from random sources alone. Even running the same provider twice with `temperature=0.7` could produce a 2.5pp swing.

That said, there are several non-stochastic reasons two "DeepSeek-V3.2" endpoints can drift, and we already see hard evidence of one of them:

1. **Sampling stochasticity dominates** (T=0.7). Two runs of the same model on the same prompt rarely give byte-identical outputs. We measured: 54% of claims got a different number of decomposed facts between the two providers — that's the same kind of variance you'd see between two independent runs of one provider.
2. **"DeepSeek-V3.2" is a public name, not a checkpoint**. Different hosts can serve different snapshots, different quantizations (FP16 vs INT8 vs AWQ), or different fine-tune variants under the same advertised name.
3. **Default sampling params at the gateway**. Even when we send `temperature=0.7`, the gateway may apply its own defaults for `top_p`, `top_k`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`. A 0.95 vs 1.0 `top_p` difference is enough to drift outputs.
4. **System prompt injection**. Many MaaS proxies silently prepend their own system prompt ("You are a helpful assistant…", safety guidelines, JSON-format hints, etc.). We pass `system_prompt=None`, but the gateway can still add one.
5. **Chat-template handling**. The conversion from OpenAI-style `messages` to DeepSeek's chat template (SP/BOS/EOS markers, role tokens) is implementation-dependent. Small template differences alter the first-token distribution.
6. **Stop sequences / max_tokens enforcement.** Slightly different truncation can chop a JSON's closing brace and force the parser into fallback modes.
7. **Retry / timeout behavior**. Vendor A may retry with a different seed on a 5xx; vendor B may surface the error. The set of "successful first-try" responses differs.

(1) is sufficient to explain 2.5pp on its own at N=200. (2)–(7) likely contribute, but on this dataset we can't separate their individual signals.

---

## 6. Recommendation

If you want a robust verdict on whether the two endpoints behave identically:

- Re-run the **same provider twice** (different RNG seed if available, or just rerun) and measure the within-provider variance. That gives the noise floor.
- Lower temperature to `0.0` (greedy) and rerun both. That removes (1) and isolates (2)–(5).
- Increase N (the 4071-claim EXFEVER run already in the repo is a much better statistical base).

At N=200 with T=0.7, the honest reading is: **qc and bit are statistically indistinguishable on the main metric**, agree on 86.5% of predictions, and exhibit the same ablation hierarchy on all the variants with effect size ≥ 2pp. The small-magnitude ablation sign flips (`wo_constraint_full`, `flip_critical`, `structural_critical`) are inside the noise floor and shouldn't be over-interpreted.

---

## 7. Files used

- qc_plan7.0 eval: `data/HOVER_subset/qc_plan7.0/dev_2_eval_by_hop_0_200.json`
- bit_plan7.0 eval: `data/HOVER_subset/bit_plan7.0/dev_2_eval_by_hop_0_200.json`
- qc_plan7.0 ablation summary: `data/HOVER_subset/qc_plan7.0/ablations_0_200/ablation_summary.md`
- bit_plan7.0 ablation summary: `data/HOVER_subset/bit_plan7.0/ablations_0_200/ablation_summary.md`
- Intermediate: `dev_2_{decomposed,questions,answers,verify,aggregate}_0_200.json` under each plan directory.
