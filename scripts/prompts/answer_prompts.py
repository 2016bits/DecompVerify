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
5. If the question is yes/no:
   - use "Yes" or "No" when the evidence is clear.
6. If the question is WH:
   - return the shortest grounded answer phrase.
7. Add bindings_update only when the answer identifies a placeholder value such as ?team, ?film, ?person, ?city, ?village, ?network, ?actor, ?stadium, ?composer, ?magazine, ?animal.
8. If no variable is resolved, return an empty object for bindings_update.
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

answer_examples = """Examples:

Example 1:
Claim:
"The capacity of the stadium that is the home ground of the team that Aamer Iqbal played for is 27,000."

Evidence:
"Hafiz Aamer Iqbal ... played twice for Lahore Eagles in List A cricket. The Lahore Eagles ... its home ground is Gaddafi Stadium. The stadium has a capacity of 27,000 spectators."

Atomic fact:
{
  "id": "f1",
  "text": "Aamer Iqbal played for ?team",
  "rely_on": [],
  "constraint": {
    "negation": null,
    "time": [],
    "quantity": []
  }
}

Question item:
{
  "fact_id": "f1",
  "main_question": "Which team did Aamer Iqbal play for?",
  "question_type": "entity_wh"
}

Current bindings:
{}

Selected question:
Which team did Aamer Iqbal play for?

Output:
{
  "fact_id": "f1",
  "answer": "Lahore Eagles",
  "status": "supported",
  "evidence_span": "played twice for Lahore Eagles in List A cricket",
  "bindings_update": {
    "?team": "Lahore Eagles"
  }
}

Example 2:
Claim:
"The capacity of the stadium that is the home ground of the team that Aamer Iqbal played for is 27,000."

Evidence:
"Hafiz Aamer Iqbal ... played twice for Lahore Eagles in List A cricket. The Lahore Eagles ... its home ground is Gaddafi Stadium."

Atomic fact:
{
  "id": "f2",
  "text": "?team has a home ground ?stadium",
  "rely_on": ["f1"],
  "constraint": {
    "negation": null,
    "time": [],
    "quantity": []
  }
}

Question item:
{
  "fact_id": "f2",
  "main_question": "What is the home ground of Lahore Eagles?",
  "question_type": "entity_wh"
}

Current bindings:
{
  "?team": "Lahore Eagles"
}

Selected question:
What is the home ground of Lahore Eagles?

Output:
{
  "fact_id": "f2",
  "answer": "Gaddafi Stadium",
  "status": "supported",
  "evidence_span": "its home ground is Gaddafi Stadium",
  "bindings_update": {
    "?stadium": "Gaddafi Stadium"
  }
}

Example 3:
Claim:
"Lady and the Tramp is an animated erotic thriller released in 1955."

Evidence:
"Lady and the Tramp is a 1955 American animated romantic musical comedy-drama film."

Atomic fact:
{
  "id": "f2",
  "text": "Lady and the Tramp is an erotic thriller",
  "rely_on": [],
  "constraint": {
    "negation": null,
    "time": [],
    "quantity": []
  }
}

Question item:
{
  "fact_id": "f2",
  "main_question": "Is Lady and the Tramp an erotic thriller?",
  "question_type": "relation_yesno"
}

Current bindings:
{}

Selected question:
Is Lady and the Tramp an erotic thriller?

Output:
{
  "fact_id": "f2",
  "answer": "No",
  "status": "contradicted",
  "evidence_span": "Lady and the Tramp is a 1955 American animated romantic musical comedy-drama film",
  "bindings_update": {}
}

Example 4:
Claim:
"The village where MKM Aboobakar was born is a village in the Indian town Tharangambadi."

Evidence:
"Thirukkalachery is a small village in Nagapattinam District, Tranquebar (Tharangambadi) Taluk, Tamil Nadu, India. Menamuna Kassim Mohamed Aboobakar ... born ... in Thirukkalachery."

Atomic fact:
{
  "id": "f1",
  "text": "MKM Aboobakar was born in ?village",
  "rely_on": [],
  "constraint": {
    "negation": null,
    "time": [],
    "quantity": []
  }
}

Question item:
{
  "fact_id": "f1",
  "main_question": "Where was MKM Aboobakar born?",
  "question_type": "entity_wh"
}

Current bindings:
{}

Selected question:
Where was MKM Aboobakar born?

Output:
{
  "fact_id": "f1",
  "answer": "Thirukkalachery",
  "status": "supported",
  "evidence_span": "born 30 April 1937 in Thirukkalachery",
  "bindings_update": {
    "?village": "Thirukkalachery"
  }
}
"""

def get_answer_prompt(claim, evidence, atomic_fact, question_item, bindings, question):
    prompt = answer_instruct + "\n\n"
    prompt += answer_examples + "\n\n"
    prompt += "Now solve the following.\n\n"
    prompt += f"Claim:\n\"{claim}\"\n\n"
    prompt += f"Evidence:\n\"{evidence}\"\n\n"
    prompt += f"Atomic fact:\n{json.dumps(atomic_fact, indent=2, ensure_ascii=False)}\n\n"
    prompt += f"Question item:\n{json.dumps(question_item, indent=2, ensure_ascii=False)}\n\n"
    prompt += f"Current bindings:\n{json.dumps(bindings, indent=2, ensure_ascii=False)}\n\n"
    prompt += f"Selected question:\n{question}\n\n"
    prompt += "Return JSON only."
    return prompt