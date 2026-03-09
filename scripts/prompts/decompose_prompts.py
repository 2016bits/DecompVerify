decompose_instruct = """You are an expert fact-checking planner.

Decompose the claim into:
1. atomic facts
2. constraints

Requirements:
- Do not extract keywords.
- Each atomic fact must contain exactly one main relation.
- Atomic facts must be minimal and independently verifiable.
- If an intermediate entity is needed, use a variable such as ?x.
- Use depends_on to show dependencies between atomic facts.
- Put negation, comparison, time, and quantity into constraints.
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

def get_decompose_prompt(claim):
    prompt = decompose_instruct + decompose_examples + f'\nClaim: "{claim}"\nDecomposition:'
    return prompt