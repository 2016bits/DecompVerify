import json

decompose_instruct = """You are an expert fact decomposition assistant for multi-hop fact verification.

Your task:
Decompose a claim into atomic facts.

Requirements:
1. Each atomic fact must express exactly one independently falsifiable proposition.
2. Every claim component that can independently make the whole claim false must be covered by at least one atomic fact.
3. Pay special attention to coverage for:
   - negation
   - time
   - quantity
   - comparison / equality / ordering
   - title / role modifiers
   - exact names
   - unique or disambiguating modifiers
4. Use rely_on to represent dependency between facts.
5. Use entity_slots to resolve cross-sentence or cross-clause coreference early.
   - If two mentions refer to the same entity, bind them to the same ?slot.
   - Replace pronouns or shorthand descriptions in atomic facts with the explicit ?slot.
6. Add a constraint field with:
   - negation: true or null
   - time: list[str]
   - quantity: list[str]
   - comparison: list[str]
   - location: list[str]
   - title_role: list[str]
   - exact_name: list[str]
   - unique_modifier: list[str]
7. Add a coverage field for each atomic fact.
   - coverage is a list of {"type": "...", "value": "..."}
   - Every independently falsifiable component should appear in at least one fact's coverage list.
8. Add a critical field for each atomic fact.
   - critical = true for each coordinated conjunct that matters to the whole claim.
   - critical = true for each unique or disambiguating qualifier.
   - critical = true for negation, time, quantity, comparison, and key title / role constraints.
   - critical = false only when dropping the fact would not change the final label.

Output JSON only.

Output schema:
{
  "claim": "...",
  "decomposition_type": "simple|coordinated|nested|comparison",
  "entity_slots": {
    "?entity_1": {
      "value": "...",
      "mentions": ["...", "..."]
    }
  },
  "atomic_facts": [
    {
      "id": "f1",
      "text": "...",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": [],
        "comparison": [],
        "location": [],
        "title_role": [],
        "exact_name": [],
        "unique_modifier": []
      },
      "coverage": [
        {"type": "exact_name", "value": "..."}
      ],
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
  "entity_slots": {},
  "atomic_facts": [
    {
      "id": "f1",
      "text": "Cotton Mather was politically influential in New England",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": [],
        "comparison": [],
        "location": ["New England"],
        "title_role": [],
        "exact_name": ["Cotton Mather", "New England", "US"],
        "unique_modifier": []
      },
      "coverage": [
        {"type": "exact_name", "value": "Cotton Mather"},
        {"type": "location", "value": "New England"}
      ],
      "critical": false
    },
    {
      "id": "f2",
      "text": "New England is a geographical region in the US",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": [],
        "comparison": [],
        "location": ["US"],
        "title_role": [],
        "exact_name": ["New England", "US"],
        "unique_modifier": []
      },
      "coverage": [
        {"type": "exact_name", "value": "New England"},
        {"type": "exact_name", "value": "US"}
      ],
      "critical": false
    },
    {
      "id": "f3",
      "text": "New England is the largest geographical region in the US",
      "rely_on": ["f2"],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": [],
        "comparison": ["largest"],
        "location": ["US"],
        "title_role": [],
        "exact_name": ["New England", "US"],
        "unique_modifier": ["largest geographical region"]
      },
      "coverage": [
        {"type": "comparison", "value": "largest"},
        {"type": "unique_modifier", "value": "largest geographical region"}
      ],
      "critical": true
    }
  ]
}

Example 2:
Claim:
"The documentaries The Truth According to Wikipedia and the 2015 film co-produced by Mary Anne Franks and Al Jazeera America do not have the same director."

Output:
{
  "claim": "The documentaries The Truth According to Wikipedia and the 2015 film co-produced by Mary Anne Franks and Al Jazeera America do not have the same director.",
  "decomposition_type": "comparison",
  "entity_slots": {
    "?doc_1": {
      "value": "The Truth According to Wikipedia",
      "mentions": ["The Truth According to Wikipedia"]
    },
    "?film_1": {
      "value": "the 2015 film co-produced by Mary Anne Franks and Al Jazeera America",
      "mentions": ["the 2015 film co-produced by Mary Anne Franks and Al Jazeera America"]
    }
  },
  "atomic_facts": [
    {
      "id": "f1",
      "text": "?doc_1 has a director",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": [],
        "comparison": [],
        "location": [],
        "title_role": ["director"],
        "exact_name": ["The Truth According to Wikipedia"],
        "unique_modifier": []
      },
      "coverage": [
        {"type": "title_role", "value": "director"},
        {"type": "exact_name", "value": "The Truth According to Wikipedia"}
      ],
      "critical": true
    },
    {
      "id": "f2",
      "text": "?film_1 has a director",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": ["2015"],
        "quantity": [],
        "comparison": [],
        "location": [],
        "title_role": ["director"],
        "exact_name": ["Mary Anne Franks", "Al Jazeera America"],
        "unique_modifier": ["co-produced by Mary Anne Franks and Al Jazeera America"]
      },
      "coverage": [
        {"type": "time", "value": "2015"},
        {"type": "exact_name", "value": "Mary Anne Franks"},
        {"type": "exact_name", "value": "Al Jazeera America"},
        {"type": "title_role", "value": "director"},
        {"type": "unique_modifier", "value": "co-produced by Mary Anne Franks and Al Jazeera America"}
      ],
      "critical": true
    },
    {
      "id": "f3",
      "text": "The director of ?doc_1 is not the same as the director of ?film_1",
      "rely_on": ["f1", "f2"],
      "constraint": {
        "negation": true,
        "time": [],
        "quantity": [],
        "comparison": ["same director"],
        "location": [],
        "title_role": ["director"],
        "exact_name": [],
        "unique_modifier": ["same director"]
      },
      "coverage": [
        {"type": "negation", "value": "do not"},
        {"type": "comparison", "value": "same director"},
        {"type": "title_role", "value": "director"}
      ],
      "critical": true
    }
  ]
}

Example 3:
Claim:
"Sean Hayes hosted the 64th Annual Tony Awards. He is best known for playing Jack McFarland."

Output:
{
  "claim": "Sean Hayes hosted the 64th Annual Tony Awards. He is best known for playing Jack McFarland.",
  "decomposition_type": "nested",
  "entity_slots": {
    "?person_1": {
      "value": "Sean Hayes",
      "mentions": ["Sean Hayes", "He"]
    }
  },
  "atomic_facts": [
    {
      "id": "f1",
      "text": "?person_1 hosted the 64th Annual Tony Awards",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": [],
        "comparison": [],
        "location": [],
        "title_role": ["host"],
        "exact_name": ["Sean Hayes", "64th Annual Tony Awards"],
        "unique_modifier": []
      },
      "coverage": [
        {"type": "exact_name", "value": "Sean Hayes"},
        {"type": "title_role", "value": "hosted the 64th Annual Tony Awards"}
      ],
      "critical": true
    },
    {
      "id": "f2",
      "text": "?person_1 is best known for playing Jack McFarland",
      "rely_on": [],
      "constraint": {
        "negation": null,
        "time": [],
        "quantity": [],
        "comparison": [],
        "location": [],
        "title_role": ["best known role"],
        "exact_name": ["Sean Hayes", "Jack McFarland"],
        "unique_modifier": ["best known"]
      },
      "coverage": [
        {"type": "exact_name", "value": "Jack McFarland"},
        {"type": "title_role", "value": "best known for playing Jack McFarland"},
        {"type": "unique_modifier", "value": "best known"}
      ],
      "critical": true
    }
  ]
}
"""


def get_decompose_prompt(claim: str, feedback: str = "") -> str:
    prompt = decompose_instruct + "\n\n"
    prompt += decompose_examples + "\n\n"
    prompt += f'Claim:\n"{claim}"\n\n'
    if feedback:
        prompt += f"Coverage / validation feedback from the previous attempt:\n{feedback}\n\n"
    prompt += "Return JSON only."
    return prompt
