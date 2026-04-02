import json

question_instruct = """
You are an expert planner for multi-hop fact verification.

Task:
Convert a decomposed claim into verification questions.

Important policy:
1. Each atomic fact gets exactly one main question.
2. The main question should target the core relation of the atomic fact.
3. If the atomic fact has a time constraint, prefer a WH time question as the main question.
4. If the atomic fact has a quantity constraint, prefer a WH quantity question as the main question.
5. For negation, do NOT force a WH question. Use a normal positive or direct verification question.
6. constraint_questions should be optional and usually empty.
7. Only generate constraint_questions for extra time/quantity clarification when clearly useful.
8. Do not generate extra constraint questions for negation.
9. Keep wording close to the original atomic fact.
10. Return JSON only.

Allowed question_type values:
- relation_yesno
- entity_wh
- time_wh
- quantity_wh
- description_wh

Output schema:
{
  "claim": "...",
  "decomposition_type": "...",
  "question_items": [
    {
      "fact_id": "f1",
      "main_question": "... ?",
      "question_type": "relation_yesno|entity_wh|time_wh|quantity_wh|description_wh",
      "constraint_questions": [
        {
          "type": "time|quantity",
          "question": "... ?"
        }
      ],
      "search_hints": ["...", "..."]
    }
  ]
}
"""

question_examples = r'''
Example 1:
Input decomposition:
{
  "claim": "The capacity of the stadium that is the home ground of the team that Aamer Iqbal played for is 27,000.",
  "decomposition_type": "quantified_or_comparative",
  "atomic_facts": [
    {
      "id": "f1",
      "text": "Aamer Iqbal played for ?team",
      "rely_on": [],
      "constraint": {"negation": null, "time": [], "quantity": []}
    },
    {
      "id": "f2",
      "text": "?team has a home ground ?stadium",
      "rely_on": ["f1"],
      "constraint": {"negation": null, "time": [], "quantity": []}
    },
    {
      "id": "f3",
      "text": "?stadium has a capacity",
      "rely_on": ["f2"],
      "constraint": {"negation": null, "time": [], "quantity": ["27,000"]}
    }
  ]
}

Output:
{
  "claim": "The capacity of the stadium that is the home ground of the team that Aamer Iqbal played for is 27,000.",
  "decomposition_type": "quantified_or_comparative",
  "question_items": [
    {
      "fact_id": "f1",
      "main_question": "What team did Aamer Iqbal play for?",
      "question_type": "entity_wh",
      "constraint_questions": [],
      "search_hints": ["Aamer Iqbal played for team", "Aamer Iqbal cricketer team"]
    },
    {
      "fact_id": "f2",
      "main_question": "What is the home ground of ?team?",
      "question_type": "entity_wh",
      "constraint_questions": [],
      "search_hints": ["?team home ground", "?team stadium"]
    },
    {
      "fact_id": "f3",
      "main_question": "What is the capacity of ?stadium?",
      "question_type": "quantity_wh",
      "constraint_questions": [],
      "search_hints": ["?stadium capacity", "?stadium 27000"]
    }
  ]
}

Example 2:
Input decomposition:
{
  "claim": "The thesis supervisor of an individual was awarded the Eddington Medal in spring 1969.",
  "decomposition_type": "nested",
  "atomic_facts": [
    {
      "id": "f1",
      "text": "The thesis supervisor of ?person was awarded the Eddington Medal",
      "rely_on": [],
      "constraint": {"negation": null, "time": ["spring 1969"], "quantity": []}
    }
  ]
}

Output:
{
  "claim": "The thesis supervisor of an individual was awarded the Eddington Medal in spring 1969.",
  "decomposition_type": "nested",
  "question_items": [
    {
      "fact_id": "f1",
      "main_question": "When was the thesis supervisor of ?person awarded the Eddington Medal?",
      "question_type": "time_wh",
      "constraint_questions": [],
      "search_hints": ["thesis supervisor Eddington Medal", "Eddington Medal 1969 thesis supervisor"]
    }
  ]
}

Example 3:
Input decomposition:
{
  "claim": "The same person did not make Shore Things and Air Force Incorporated.",
  "decomposition_type": "simple",
  "atomic_facts": [
    {
      "id": "f1",
      "text": "The same person made Shore Things and Air Force Incorporated",
      "rely_on": [],
      "constraint": {"negation": true, "time": [], "quantity": []}
    }
  ]
}

Output:
{
  "claim": "The same person did not make Shore Things and Air Force Incorporated.",
  "decomposition_type": "simple",
  "question_items": [
    {
      "fact_id": "f1",
      "main_question": "Did the same person make Shore Things and Air Force Incorporated?",
      "question_type": "relation_yesno",
      "constraint_questions": [],
      "search_hints": ["Shore Things director", "Air Force Incorporated director"]
    }
  ]
}
'''

question_repair_prompt = """
You are an expert checker for multi-hop fact verification question planning.

You are given:
1. a claim
2. a decomposition result
3. a current question plan
4. detected issues

Repair checklist:
1. Each atomic fact must have exactly one main question.
2. If a fact has a time constraint, prefer a WH time question as the main question.
3. If a fact has a quantity constraint, prefer a WH quantity question as the main question.
4. For negation, do not force WH questions.
5. constraint_questions should usually be empty.
6. Only keep optional extra constraint questions for time or quantity.
7. search_hints should be concise and useful for retrieval.
8. Return JSON only.

claim: [CLAIM]
decomposition: [DECOMPOSITION]
current question plan: [QUESTION_PLAN]
issues:
[ISSUES]

Return JSON only in this schema:
{
  "claim": "...",
  "decomposition_type": "...",
  "question_items": [
    {
      "fact_id": "f1",
      "main_question": "... ?",
      "question_type": "relation_yesno|entity_wh|time_wh|quantity_wh|description_wh",
      "constraint_questions": [
        {
          "type": "time|quantity",
          "question": "... ?"
        }
      ],
      "search_hints": ["...", "..."]
    }
  ]
}
"""


def get_question_prompt(claim, decomposition):
    return (
        question_instruct
        + "\n"
        + question_examples
        + "\nClaim: "
        + json.dumps(claim, ensure_ascii=False)
        + "\nDecomposition:\n"
        + json.dumps(decomposition, indent=2, ensure_ascii=False)
        + "\nQuestion plan:"
    )


def get_question_repair_prompt(claim, decomposition, question_plan, issues):
    return (
        question_repair_prompt.replace('[CLAIM]', claim)
        .replace('[DECOMPOSITION]', json.dumps(decomposition, indent=2, ensure_ascii=False))
        .replace('[QUESTION_PLAN]', json.dumps(question_plan, indent=2, ensure_ascii=False))
        .replace('[ISSUES]', '\n'.join(f'- {issue}' for issue in issues))
    )
