import json


decompose_instruct = """
You are an expert planner for multi-hop fact verification.

Task:
Decompose a claim into minimal atomic facts for step-wise verification.

Core principle:
Each atomic fact must express exactly ONE core verifiable relation.

Allowed decomposition types:
- simple
- coordinated
- nested
- quantified_or_comparative

Constraint policy:
Only keep these three constraint types:
- negation
- time
- quantity

Important output rules:
- Return VALID JSON only.
- Do not output markdown.
- Do not output explanations.
- Use only this schema.
- Use ids f1, f2, f3, ...
- rely_on may only contain earlier fact ids.
- negation must be true or null.
- time must be a list.
- quantity must be a list.

Output schema:
{
  "claim": "...",
  "decomposition_type": "simple",
  "atomic_facts": [
    {
      "id": "f1",
      "text": "...",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      }
    }
  ]
}
"""


decompose_examples = r'''
Example 1:
{
  "claim": "Marie Curie was born in Warsaw and won two Nobel Prizes.",
  "decomposition_type": "coordinated",
  "atomic_facts": [
    {
      "id": "f1",
      "text": "Marie Curie was born in Warsaw",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      }
    },
    {
      "id": "f2",
      "text": "Marie Curie won Nobel Prizes",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": ["two"]
      }
    }
  ]
}

Example 2:
{
  "claim": "The director of the film starring Tom Hanks was born in California.",
  "decomposition_type": "nested",
  "atomic_facts": [
    {
      "id": "f1",
      "text": "Tom Hanks starred in ?film",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      }
    },
    {
      "id": "f2",
      "text": "?film was directed by ?director",
      "rely_on": ["f1"],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      }
    },
    {
      "id": "f3",
      "text": "?director was born in California",
      "rely_on": ["f2"],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      }
    }
  ]
}
'''


decompose_repair_prompt = """
You are an expert checker for multi-hop fact decomposition.

You are given:
1. a claim
2. a current decomposition result
3. detected issues

Repair the decomposition so that it is faithful to the claim and suitable for step-wise verification.

Repair rules:
- Return VALID JSON only.
- Do not output markdown.
- Do not output explanations.
- Use one of these decomposition types only:
  - simple
  - coordinated
  - nested
  - quantified_or_comparative
- Each atomic fact must only contain:
  - id
  - text
  - rely_on
  - constraint
- constraint must only contain:
  - negation
  - time
  - quantity
- negation must be true or null.
- time must be a list.
- quantity must be a list.
- rely_on may only reference earlier fact ids.

claim: [CLAIM]
current decomposition: [DECOMPOSITION]
issues:
[ISSUES]

Return JSON only in this schema:
{
  "claim": "...",
  "decomposition_type": "simple",
  "atomic_facts": [
    {
      "id": "f1",
      "text": "...",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      }
    }
  ]
}
"""


def get_decompose_prompt(claim: str) -> str:
    return decompose_instruct + "\n" + decompose_examples + f'\nClaim: "{claim}"\nReturn JSON only:'



def get_repair_prompt(claim, decomposition, issues):
    return (
        decompose_repair_prompt.replace('[CLAIM]', claim)
        .replace('[DECOMPOSITION]', json.dumps(decomposition, indent=2, ensure_ascii=False))
        .replace('[ISSUES]', '\n'.join(f'- {issue}' for issue in issues))
    )
