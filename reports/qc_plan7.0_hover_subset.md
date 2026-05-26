# qc_plan7.0 on HOVER_subset — Results Summary

**Setup**

- Plan: `qc_plan7.0` (DeepSeek-V3.2 via aiping endpoint)
- Dataset: `HOVER_subset`, `dev` split, **200 claims**
- Label setting: binary (NEI merged into refutes), `class_num=2`
- Range: `0_200`, `max_workers=2`
- Source scripts: `scripts/run_scripts/run_DeepSeek.sh`, `scripts/run_scripts/run_ablation.sh`

---

## 1. Main pipeline (`run_DeepSeek.sh`)

Source: `data/HOVER_subset/qc_plan7.0/dev_2_eval_by_hop_0_200.json`

### Overall

| metric | value |
|---|---:|
| accuracy | **0.785** (157/200) |
| macro precision | 0.7851 |
| macro recall | 0.7850 |
| **macro F1** | **0.7850** |
| supports F1 | 0.7839 (P 0.7879 / R 0.78, TP 78 / FP 21 / FN 22) |
| refutes F1 | 0.7861 (P 0.7822 / R 0.79, TP 79 / FP 22 / FN 21) |
| pred distribution | refutes 101 / supports 99 (gold: 100/100) |

### By hop

| hop | N | accuracy | macro F1 | notes |
|---|---:|---:|---:|---|
| 2 | 57 | **0.807** | 0.807 | strongest; refutes recall 0.7419 |
| 3 | 91 | 0.780 | 0.780 | bulk of the data, balanced |
| 4 | 52 | 0.769 | 0.768 | hardest; supports recall drops to 0.6923 |

### Confusion matrix

|  | pred supports | pred refutes |
|---|---:|---:|
| gold supports | 78 | 22 |
| gold refutes | 21 | 79 |

Errors are nearly symmetric across the two classes.

---

## 2. Ablation suite (`run_ablation.sh`)

Source: `data/HOVER_subset/qc_plan7.0/ablations_0_200/ablation_summary.md`

Baseline (`full`) = **0.7850 acc / 0.7850 macro F1**. Sorted by harm to overall macro F1.

| variant | acc | Δ macro F1 | notes |
|---|---:|---:|---|
| **full** (baseline) | 0.7850 | — | reference |
| random_critical | 0.7850 | +0.0000 | critical-gate randomization had no observable effect at N=200 |
| no_critical_gate | 0.7850 | +0.0000 | same |
| wo_rely_on_aggregate_only | 0.7850 | +0.0000 | dependency-marker removal at aggregate side: no flips |
| decomp_no_aggregation | 0.7850 | +0.0000 | same |
| wo_runtime_binding | 0.7800 | -0.0050 | 1 flip only |
| wo_constraint_full | 0.7750 | -0.0100 | overall −1pp; but hop2 +1.75pp, hop4 −3.87pp |
| structural_critical | 0.7750 | -0.0100 | 2 flips, both wrong |
| **wo_adjudication** | 0.7600 | **-0.0262** | 2nd-pass adjudication helps; 13 flips, net −5 |
| **flip_critical** | 0.7600 | -0.0254 | reversing critical markers under-predicts refutes |
| **wo_value_checks** | 0.7450 | **-0.0414** | value constraints carry the most weight on hop3/hop4 |
| **wo_polarity** | 0.7450 | **-0.0410** | polarity/negation is equally critical; hop2 −5.3pp |
| **majority_vote** | **0.5750** | **-0.2796** | catastrophic; 175/25 collapse toward supports, 76 flips, net −42 |

`naive_single_shot` was not run (`RUN_NAIVE_SINGLE_SHOT=0` by default).

### Per-hop macro F1 deltas (selected)

| variant | hop2 | hop3 | hop4 |
|---|---:|---:|---:|
| full | 0.8070 | 0.7800 | 0.7678 |
| wo_value_checks | 0.7863 (−0.021) | 0.7251 (−0.055) | 0.7271 (−0.041) |
| wo_polarity | 0.7537 (−0.053) | 0.7471 (−0.033) | 0.7243 (−0.044) |
| wo_adjudication | 0.7537 (−0.053) | 0.7691 (−0.011) | 0.7423 (−0.026) |
| flip_critical | 0.7171 (−0.090) | 0.7575 (−0.022) | 0.8074 (+0.040) |
| majority_vote | 0.5129 (−0.294) | 0.4997 (−0.280) | 0.5034 (−0.264) |

---

## 3. Key findings

1. Baseline lands at **78.5% acc / 0.785 macro F1**, with a class-symmetric error profile and almost no prior bias.
2. **Accuracy decreases monotonically with hop count** (80.7 → 78.0 → 76.9), consistent with multi-hop difficulty.
3. Module importance (largest to smallest harm when removed):
   `majority_vote replacement ≫ value_checks ≈ polarity > adjudication ≈ critical-flip > constraint_full > runtime_binding`.
4. Four variants produced **zero change** at N=200:
   `random_critical`, `no_critical_gate`, `wo_rely_on_aggregate_only`, `decomp_no_aggregation`.
   Either the corresponding mechanism is dormant on this baseline path, or N=200 is too small to surface differences.
5. The critical-marker family (group 1) only shows a meaningful drop when **direction is reversed** (`flip_critical`, −2.5pp); merely turning the gate off or randomizing it is neutral here.
6. `wo_constraint_full` is a net negative overall, but is **+1.75pp on hop2** while **−3.87pp on hop4** — constraints can mildly over-fit short hops yet remain useful for long hops.

---

## 4. Result file index

- Main pipeline eval: `data/HOVER_subset/qc_plan7.0/dev_2_eval_by_hop_0_200.json`
- Ablation summary: `data/HOVER_subset/qc_plan7.0/ablations_0_200/ablation_summary.{md,csv,json}`
- Per-variant outputs: `data/HOVER_subset/qc_plan7.0/ablations_0_200/<variant>_{eval,aggregate,verify}.json`
- Intermediate pipeline artefacts: `data/HOVER_subset/qc_plan7.0/dev_2_{decomposed,questions,answers,verify,aggregate}_0_200.json`
