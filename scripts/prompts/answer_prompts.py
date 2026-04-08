import json

answer_instruct = """You are an expert fact-checking assistant.

You are given:
1. a claim
2. an evidence passage
3. one atomic fact
4. one question item
5. the current variable bindings
6. one selected question

Your task:
Answer the selected question using the evidence only.
If the answer resolves a variable placeholder, add it to bindings_update.

Rules:
1. Use only the provided evidence.
2. Return JSON only.
3. Keep the answer short.
4. If the evidence does not clearly answer the question, return:
   - "answer": "insufficient"
   - "status": "insufficient"
5. If question_type is relation_yesno:
   - answer with Yes or No only.
6. If question_type is entity_wh:
   - return the shortest grounded answer phrase.
   - if answer_slot is provided, fill bindings_update with that slot.
7. If question_type is time_wh:
   - return the date or year only.
8. If question_type is quantity_wh:
   - return the quantity only.
9. Copy a short evidence span.

Output schema:
{
  "fact_id": "f1",
  "answer": "...",
  "status": "supported|contradicted|insufficient",
  "evidence_span": "...",
  "bindings_update": {}
}
"""

def get_answer_prompt(claim, evidence, atomic_fact, question_item, bindings, question):
    prompt = answer_instruct + "\n\n"
    prompt += f'Claim:\n"{claim}"\n\n'
    prompt += f'Evidence:\n"{evidence}"\n\n'
    prompt += f"Atomic fact:\n{json.dumps(atomic_fact, indent=2, ensure_ascii=False)}\n\n"
    prompt += f"Question item:\n{json.dumps(question_item, indent=2, ensure_ascii=False)}\n\n"
    prompt += f"Current bindings:\n{json.dumps(bindings, indent=2, ensure_ascii=False)}\n\n"
    prompt += f"Selected question:\n{question}\n\n"
    prompt += "Return JSON only."
    return prompt