import json

question_instruct = """You are an expert question-planning assistant for multi-hop fact verification.

Given a claim decomposition, generate one verification-oriented main question for each atomic fact.

Rules:
1. Keep exactly one main question per atomic fact.
2. Choose question_type from:
   - relation_yesno
   - entity_wh
   - time_wh
   - quantity_wh
3. For entity_wh questions, provide answer_slot when the question is expected to resolve a variable-like entity.
4. Return JSON only.

Output schema:
{
  "question_items": [
    {
      "fact_id": "f1",
      "main_question": "...",
      "question_type": "entity_wh",
      "answer_slot": "?team",
      "constraint_questions": [],
      "search_hints": ["..."]
    }
  ]
}
"""

def get_question_prompt(claim, decomposition):
    prompt = question_instruct + "\n\n"
    prompt += f'Claim:\n"{claim}"\n\n'
    prompt += f"Decomposition:\n{json.dumps(decomposition, indent=2, ensure_ascii=False)}\n\n"
    prompt += "Return JSON only."
    return prompt