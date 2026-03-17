import json

decompose_instruct = """You are an expert fact-checking planner.

Decompose the claim into:
1. atomic facts
2. constraints

Requirements:
- Do not extract keywords.
- Each atomic fact must contain exactly one main relation.
- Atomic facts must be minimal, faithful to the claim, and independently verifiable.
- Prefer the fewest atomic facts that still preserve the claim's meaning.
- Do not over-decompose.
- If an intermediate entity is needed for later verification, use a variable such as ?x.
- Use depends_on only when a later fact truly needs a variable resolved by an earlier fact.
- Do not add a bridge fact unless it is necessary for a later verify fact or constraint.
- Put negation, comparison, time, and quantity into constraints whenever possible.
- Keep attachment scope correct:
  - relative clauses
  - prepositional phrases
  - appositives
  - temporal modifiers
- Do not add external knowledge.
- Return JSON only.

Output schema:
{
  "claim": "...",
  "atomic_facts": [
    {
      "id": "f1",
      "subject": "...",
      "predicate": "...",
      "object": "...",
      "role": "bridge|verify|anchor",
      "depends_on": []
    }
  ],
  "constraints": [
    {
      "type": "negation|comparison|time|quantity",
      "target_facts": ["f1"],
      "operator": "not|before|after|earlier_than|later_than|>|<|=|>=|<=",
      "value": ""
    }
  ]
}
"""

decompose_examples = """
Examples:
{
    "claim": "Robert J. O'Neill was born April 10, 1976.",
    "atomic_facts": [
      {
        "id": "f1",
        "subject": "Robert J. O'Neill",
        "predicate": "born on",
        "object": "April 10, 1976",
        "role": "verify",
        "depends_on": []
      }
    ],
    "constraints": []
  },
  {
    "claim": "The capital of France is Paris.",
    "atomic_facts": [
      {
        "id": "f1",
        "subject": "the capital of France",
        "predicate": "is",
        "object": "Paris",
        "role": "verify",
        "depends_on": []
      }
    ],
    "constraints": []
  },
  {
    "claim": "The director of the film starring Tom Hanks was born in California.",
    "atomic_facts": [
      {
        "id": "f1",
        "subject": "Tom Hanks",
        "predicate": "starred in",
        "object": "?film",
        "role": "bridge",
        "depends_on": []
      },
      {
        "id": "f2",
        "subject": "?film",
        "predicate": "directed by",
        "object": "?director",
        "role": "bridge",
        "depends_on": ["f1"]
      },
      {
        "id": "f3",
        "subject": "?director",
        "predicate": "born in",
        "object": "California",
        "role": "verify",
        "depends_on": ["f2"]
      }
    ],
    "constraints": []
  },
  {
    "claim": "Puerto Rico is not an unincorporated territory of the United States.",
    "atomic_facts": [
      {
        "id": "f1",
        "subject": "Puerto Rico",
        "predicate": "is",
        "object": "an unincorporated territory of the United States",
        "role": "verify",
        "depends_on": []
      }
    ],
    "constraints": [
      {
        "type": "negation",
        "target_facts": ["f1"],
        "operator": "not",
        "value": ""
      }
    ]
  },
  {
    "claim": "Cyndi Lauper won the Best New Artist award earlier than Adele.",
    "atomic_facts": [
      {
        "id": "f1",
        "subject": "Cyndi Lauper",
        "predicate": "won Best New Artist in",
        "object": "?t1",
        "role": "anchor",
        "depends_on": []
      },
      {
        "id": "f2",
        "subject": "Adele",
        "predicate": "won Best New Artist in",
        "object": "?t2",
        "role": "anchor",
        "depends_on": []
      }
    ],
    "constraints": [
      {
        "type": "comparison",
        "target_facts": ["f1", "f2"],
        "operator": "earlier_than",
        "value": ""
      }
    ]
  }
"""

decompose_repair_prompt = """You are an expert checker for multi-hop fact-checking plans.

You are given:
1. a claim
2. a decomposition plan with atomic facts and constraints
3. detected issues in the decomposition

Your task:
Repair the decomposition so that it is faithful to the claim and suitable for step-wise verification.

What to check:
1. Each atomic fact must faithfully reflect the claim.
2. Each atomic fact must contain exactly one main relation.
3. Dependencies must represent real answer flow:
   - fact B depends on fact A only if fact B needs a variable resolved by fact A.
4. Relative clauses or modifiers must attach to the correct entity.
5. Do not introduce facts that change the meaning of the claim.
6. Remove redundant or incorrect atomic facts.
7. Prefer fewer atomic facts when multiple decompositions are possible.
8. Keep the main semantic relation of the claim intact; do not weaken it into loosely related facts.
9. A bridge fact is valid only if it resolves a variable used later by a verify fact or constraint.
10. Keep the schema unchanged.
11. Return JSON only.

claim:
[CLAIM]

current decomposition:
[DECOMPOSITION]

issues:
[ISSUES]

Return JSON only in this schema:
{{
  "claim": "...",
  "atomic_facts": [
    {{
      "id": "f1",
      "subject": "...",
      "predicate": "...",
      "object": "...",
      "role": "bridge|verify|anchor",
      "depends_on": []
    }}
  ],
  "constraints": [
    {{
      "type": "negation|comparison|time|quantity",
      "target_facts": ["f1"],
      "operator": "not|before|after|earlier_than|later_than|>|<|=|>=|<=",
      "value": ""
    }}
  ]
}}
"""

def get_decompose_prompt(claim):
    prompt = decompose_instruct + decompose_examples + f'\nClaim: "{claim}"\nDecomposition:'
    return prompt

def get_repair_prompt(claim, decomposition, issues):
    prompt = decompose_repair_prompt.replace('[CLAIM]', claim).replace('[DECOMPOSITION]', json.dumps(decomposition, indent=4)).replace('[ISSUES]', '\n'.join(f'- {issue}' for issue in issues))
    return prompt