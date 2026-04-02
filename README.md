# DecompVerify: Dependency-aware Atomic Fact Verification for Multi-hop Fact Checking

This project implements a dependency-aware, step-wise pipeline for multi-hop fact checking.

## Overview

Instead of verifying a complex claim in one step, the system decomposes it into atomic facts, verifies them one by one, and then aggregates the results into a final claim label.

Current pipeline:

**Claim  
→ Atomic decomposition  
→ Question planning  
→ Answer generation  
→ Atomic fact verification  
→ Claim aggregation  
→ Evaluation**

The current implementation is organized around the following stages:

- `scripts/decompose_atomic_facts.py`
- `scripts/generate_question.py`
- `scripts/get_answer.py`
- `scripts/verify_atomic_facts.py`
- `scripts/aggregate_labels.py`
- `scripts/utils/evaluate_acc.py`

---

## 1. Method Summary

The current method is a **dependency-aware, modular fact-checking pipeline** designed for complex multi-hop claims.

Its core ideas are:

1. **Decompose a complex claim into atomic facts**
2. **Preserve dependencies between atomic facts**
3. **Generate verification-oriented questions**
4. **Answer each question using evidence**
5. **Verify each atomic fact with target-aware consistency checks**
6. **Aggregate local verification results with critical-fact gating**

The method explicitly models:

- internal compositional structure of claims
- multi-hop dependencies
- key constraints such as negation, time, and quantity
- critical sub-facts that can determine the final claim label

---

## 2. Atomic Decomposition

A claim is decomposed into a set of atomic facts.

Each atomic fact follows the schema:

```json
{
  "id": "f1",
  "text": "...",
  "rely_on": [],
  "constraint": {
    "negation": null,
    "time": [],
    "quantity": []
  },
  "critical": false
}
```

### Field descriptions

- `id`  
  Unique identifier of the atomic fact.

- `text`  
  Natural language representation of one core relation.

- `rely_on`  
  List of prerequisite facts that this fact depends on.

- `constraint`  
  Structured constraints extracted from the claim:
  - `negation`
  - `time`
  - `quantity`

- `critical`  
  Whether the fact is essential for determining the final truth of the whole claim.

### Decomposition principles

- Each atomic fact should express **one core relation only**.
- Coordinated structures are split into separate facts.
- Nested structures are decomposed step by step.
- Time, quantity, and negation are kept in `constraint`.
- Important modifiers that can flip the final label are marked as `critical=true`.

### Supported decomposition types

- `simple`
- `coordinated`
- `nested`
- `comparison`

---

## 3. Critical Facts

A fact is usually marked as `critical=true` if it involves:

- negation
- explicit time or quantity constraints
- comparison / equality / inequality
- superlatives / ordinals / ranking
- key location or identity constraints
- any modifier that can flip the final label

### Typical examples of critical facts

- `New England is the largest geographical region in the US`
- `Toby Williams was born in Norfolk`
- `The two films do not have the same director`
- `The stadium capacity is 27,000`
- `The battle took place in Humber of Yorkshire`

Critical facts are treated as gating conditions during final aggregation.

---

## 4. Question Planning

Each atomic fact is converted into one main question for verification.

Question item schema:

```json
{
  "fact_id": "f1",
  "fact_text": "...",
  "rely_on": [],
  "constraint": {...},
  "main_question": "...",
  "question_type": "relation_yesno | entity_wh | time_wh | quantity_wh",
  "constraint_questions": [],
  "search_hints": [...]
}
```

### Question types

- `relation_yesno`  
  Used for relation verification.

- `entity_wh`  
  Used for entity, location, director, channel, identity, etc.

- `time_wh`  
  Used for time constraints.

- `quantity_wh`  
  Used for numeric or ranking constraints.

### Goal of question planning

The goal is not just to paraphrase the fact, but to generate a question that makes downstream answering and verification reliable.

---

## 5. Answer Generation

Given:

- the claim
- the gold evidence
- one atomic fact
- one question item
- optional intermediate bindings

the model returns:

```json
{
  "fact_id": "f1",
  "question": "...",
  "answer": "...",
  "status": "supported | contradicted | insufficient | api_error",
  "evidence_span": "...",
  "bindings_update": {}
}
```

### Notes

- Answer generation is strictly evidence-grounded.
- For dependent multi-hop facts, intermediate bindings can be propagated to later questions.
- If an API failure or rate-limit failure occurs, it is marked as `api_error` rather than silently treated as `insufficient`.

---

## 6. Atomic Fact Verification

Each answer is converted into a fact-level verification label:

- `support`
- `contradict`
- `insufficient`

Verification output typically includes:

```json
{
  "fact_id": "f1",
  "fact_text": "...",
  "bound_fact_text": "...",
  "question": "...",
  "bound_question": "...",
  "question_type": "...",
  "answer": "...",
  "answer_status": "...",
  "verification_label": "support | contradict | insufficient",
  "reason": "...",
  "evidence_span": "...",
  "constraint": {...},
  "rely_on": [...],
  "critical": true
}
```

### Verification rules by question type

#### 6.1 relation_yesno
- Yes → support
- No → contradict
- If the fact is negated, the polarity is reversed.

#### 6.2 time_wh
- Compare the answer with `constraint.time`
- Match → support
- Conflict → contradict
- Otherwise → insufficient

#### 6.3 quantity_wh
- Compare the answer with `constraint.quantity`
- Match → support
- Conflict → contradict
- Otherwise → insufficient

#### 6.4 entity_wh
For `entity_wh`, the system applies an **answer-target consistency check**.

If the atomic fact contains an explicit target value, the verifier checks:

1. whether the answer is consistent with that target
2. whether the evidence span truly supports that target

Decision:

- consistent → `support`
- clearly inconsistent → `contradict`
- uncertain → `insufficient`

This prevents the system from treating any plausible entity answer as automatic support.

---

## 7. Claim Aggregation

The final claim label is determined using **critical fact gating**.

### Aggregation rules

1. If any `critical=true` fact is contradicted  
   → `refutes`

2. If any `critical=true` fact is insufficient  
   → `not enough information`

3. If any non-critical fact is contradicted  
   → `refutes`

4. `supports` is assigned only if:
   - all critical facts are supported
   - no contradicted facts exist
   - only a small number of non-critical facts remain insufficient

5. Otherwise  
   → `not enough information`

### Binary evaluation setting

For binary evaluation:

- `supports` → `supports`
- `refutes` + `not enough information` → `refutes`

This makes critical facts function as hard gates for final support decisions.

---

## 8. End-to-end Workflow

The full workflow is:

### Step 1: Decompose claim
Input:
- claim

Output:
- decomposition type
- atomic facts with `constraint` and `critical`

### Step 2: Generate question plan
Input:
- decomposition result

Output:
- one verification-oriented question per atomic fact

### Step 3: Generate answers
Input:
- claim
- gold evidence
- question plan

Output:
- answer for each atomic fact
- evidence span
- optional bindings update

### Step 4: Verify atomic facts
Input:
- decomposition
- question plan
- answer result
- gold evidence

Output:
- fact-level labels: `support / contradict / insufficient`

### Step 5: Aggregate to claim label
Input:
- verified atomic facts

Output:
- final claim label:
  - `supports`
  - `refutes`
  - `not enough information`

### Step 6: Evaluate
Input:
- aggregated results
- gold labels

Output:
- accuracy
- precision / recall / F1
- macro-F1
- by-hop F1
- confusion matrix

---

## 9. Evaluation

The project supports both claim-level and hop-level evaluation.

### Common metrics

- Accuracy
- Precision / Recall / F1
- Macro-F1
- By-hop F1
- Confusion matrix

### Additional statistics

Useful diagnostic statistics include:

- decomposition type distribution
- ratio of critical facts
- unresolved variable rate
- answer insufficient rate
- verification label distribution
- aggregation label distribution

---

## 10. Recommended Execution Order

Typical execution order:

```bash
python scripts/decompose_atomic_facts.py
python scripts/generate_question.py
python scripts/get_answer.py
python scripts/verify_atomic_facts.py
python scripts/aggregate_labels.py
python scripts/utils/evaluate_acc.py
```

---

## 11. Key Characteristics of the Current Method

The current method emphasizes:

- **atomic decomposition**
- **dependency preservation**
- **evidence-grounded local answering**
- **target-consistent fact verification**
- **critical-fact-gated aggregation**

This design is especially suitable for:

- multi-hop fact checking
- compositional claims
- claims with nested modifiers
- claims whose truth depends on one or a few crucial sub-facts

---

## 12. Summary

The current system is a dependency-aware, step-wise verification pipeline for multi-hop fact checking.

Its main strengths are:

- explicit claim decomposition
- structured local verification
- stronger handling of key constraints
- more reliable aggregation through critical fact gating

Rather than relying on simple majority voting over all sub-facts, the method explicitly models the fact that, in many real claims, **a small number of critical atomic facts determine the final label**.
