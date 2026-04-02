import json

decompose_instruct = """You are an expert fact decomposition assistant for multi-hop fact verification.

Your task:
Decompose a claim into atomic facts.

Requirements:
1. Each atomic fact should express only ONE core relation.
2. Use rely_on to represent dependency between facts.
3. Add a constraint field with:
   - negation: true or null
   - time: list[str]
   - quantity: list[str]
4. Add a critical field for each atomic fact:
   - critical = true if this fact is essential for determining the final truth of the claim
   - critical = false otherwise

Mark a fact as critical=true when it involves one of the following:
- negation
- explicit time / quantity constraints
- comparison / equality / inequality
- superlative / ordinal / ranking
- key location or identity constraints
- key modifier that can flip the final label
Examples:
- "largest", "smallest", "first", "same", "different"
- "born in Norfolk"
- "shown on Sky Living"
- "the same director"
- "not the same director"

Output JSON only.

Output schema:
{
  "claim": "...",
  "decomposition_type": "simple|coordinated|nested|comparison",
  "atomic_facts": [
    {
      "id": "f1",
      "text": "...",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      },
      "critical": true
    }
  ]
}
"""

decompose_examples = """Example 1:
Claim:
"Cotton Mather was politically influential in New England, the largest geographical region in the US."

Output:
{
  "claim": "Cotton Mather was politically influential in New England, the largest geographical region in the US.",
  "decomposition_type": "nested",
  "atomic_facts": [
    {
      "id": "f1",
      "text": "Cotton Mather was politically influential in New England",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      },
      "critical": false
    },
    {
      "id": "f2",
      "text": "New England is a geographical region in the US",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      },
      "critical": false
    },
    {
      "id": "f3",
      "text": "New England is the largest geographical region in the US",
      "rely_on": ["f2"],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      },
      "critical": true
    }
  ]
}

Example 2:
Claim:
"Toby Williams is a British actor, writer and award-winning stand-up comedian born in Norfolk who has appeared in Trying Again shown on Sky Living."

Output:
{
  "claim": "Toby Williams is a British actor, writer and award-winning stand-up comedian born in Norfolk who has appeared in Trying Again shown on Sky Living.",
  "decomposition_type": "nested",
  "atomic_facts": [
    {
      "id": "f1",
      "text": "Toby Williams is a British actor",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      },
      "critical": false
    },
    {
      "id": "f2",
      "text": "Toby Williams is a writer",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      },
      "critical": false
    },
    {
      "id": "f3",
      "text": "Toby Williams is an award-winning stand-up comedian",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      },
      "critical": false
    },
    {
      "id": "f4",
      "text": "Toby Williams was born in Norfolk",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      },
      "critical": true
    },
    {
      "id": "f5",
      "text": "Toby Williams has appeared in Trying Again",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      },
      "critical": false
    },
    {
      "id": "f6",
      "text": "Trying Again was shown on Sky Living",
      "rely_on": ["f5"],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      },
      "critical": true
    }
  ]
}

Example 3:
Claim:
"The documentaries The Truth According to Wikipedia and the 2015 film co-produced by Mary Anne Franks and Al Jazeera America do not have the same director."

Output:
{
  "claim": "The documentaries The Truth According to Wikipedia and the 2015 film co-produced by Mary Anne Franks and Al Jazeera America do not have the same director.",
  "decomposition_type": "comparison",
  "atomic_facts": [
    {
      "id": "f1",
      "text": "The Truth According to Wikipedia has a director",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": []
      },
      "critical": false
    },
    {
      "id": "f2",
      "text": "The 2015 film co-produced by Mary Anne Franks and Al Jazeera America has a director",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": ["2015"],
        "quantity": []
      },
      "critical": false
    },
    {
      "id": "f3",
      "text": "The director of The Truth According to Wikipedia is not the same as the director of the 2015 film",
      "rely_on": ["f1", "f2"],
      "constraint": {
        "negation": true,
        "time": [],
        "quantity": []
      },
      "critical": true
    }
  ]
}
"""

def get_decompose_prompt(claim: str) -> str:
    prompt = decompose_instruct + "\n\n"
    prompt += decompose_examples + "\n\n"
    prompt += f'Claim:\n"{claim}"\n\n'
    prompt += "Return JSON only."
    return prompt