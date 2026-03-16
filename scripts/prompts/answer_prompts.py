import json

answer_instruct = """You are an expert fact-checking assistant.

Given:
1. a claim
2. an evidence passage
3. one atomic fact
4. the current variable bindings
5. question templates for this atomic fact

Your task:
Answer the atomic fact using the evidence only.

Requirements:
1. Use only the provided evidence.
2. Return JSON only.
3. If the fact contains an unresolved variable, answer the question to identify that variable.
4. If the fact is fully instantiated, determine whether it is supported, contradicted, or insufficiently supported.
5. Quote a short supporting evidence span from the evidence.
6. If you identify a new variable value, put it in bindings_update.
7. If the evidence does not justify a confident answer, use "insufficient".

Output schema:
{{
  "fact_id": "f1",
  "question": "...",
  "answer": "...",
  "status": "supported|contradicted|insufficient",
  "evidence_span": "...",
  "bindings_update": {{}}
}}
"""

answer_examples = """Examples:

Example 1:

Claim:
"Jack McFarland is the best known role of the host of the 64th Annual Tony Awards."

Evidence:
"Sean Patrick Hayes is an American actor. He is best known for his role as Jack McFarland on the NBC sitcom Will & Grace. The host of the 64th Annual Tony Awards was Sean Hayes."

Atomic fact:
{
  "id": "f1",
  "subject": "?actor",
  "predicate": "played",
  "object": "Jack McFarland",
  "role": "bridge",
  "depends_on": []
}

Filled fact:
{
  "subject": "?actor",
  "predicate": "played",
  "object": "Jack McFarland"
}

Question templates:
{
  "primary": "Who played Jack McFarland?",
  "alternate": "Which actor played Jack McFarland?",
  "boolean": "",
  "boolean_template": "Did {value} play Jack McFarland?"
}

Selected question:
Who played Jack McFarland?

Output:
{
  "fact_id": "f1",
  "question": "Who played Jack McFarland?",
  "answer": "Sean Hayes",
  "status": "supported",
  "evidence_span": "He is best known for his role as Jack McFarland on the NBC sitcom Will & Grace.",
  "bindings_update": {
    "?actor": "Sean Hayes"
  }
}


Example 2:

Claim:
"Jack McFarland is the best known role of the host of the 64th Annual Tony Awards."

Evidence:
"The host of the 64th Annual Tony Awards was Sean Hayes."

Atomic fact:
{
  "id": "f2",
  "subject": "?actor",
  "predicate": "hosted",
  "object": "64th Annual Tony Awards",
  "role": "verify",
  "depends_on": ["f1"]
}

Filled fact:
{
  "subject": "Sean Hayes",
  "predicate": "hosted",
  "object": "64th Annual Tony Awards"
}

Selected question:
Did Sean Hayes host the 64th Annual Tony Awards?

Output:
{
  "fact_id": "f2",
  "question": "Did Sean Hayes host the 64th Annual Tony Awards?",
  "answer": "Yes",
  "status": "supported",
  "evidence_span": "The host of the 64th Annual Tony Awards was Sean Hayes.",
  "bindings_update": {}
}


Example 3:

Claim:
"The director of the film starring Tom Hanks was born in California."

Evidence:
"Saving Private Ryan starred Tom Hanks. The film was directed by Steven Spielberg. Steven Spielberg was born in Cincinnati, Ohio."

Atomic fact:
{
  "id": "f3",
  "subject": "?director",
  "predicate": "born in",
  "object": "California",
  "role": "verify",
  "depends_on": ["f2"]
}

Filled fact:
{
  "subject": "Steven Spielberg",
  "predicate": "born in",
  "object": "California"
}

Selected question:
Was Steven Spielberg born in California?

Output:
{
  "fact_id": "f3",
  "question": "Was Steven Spielberg born in California?",
  "answer": "No",
  "status": "contradicted",
  "evidence_span": "Steven Spielberg was born in Cincinnati, Ohio.",
  "bindings_update": {}
}


Example 4:

Claim:
"Cyndi Lauper won the Best New Artist award earlier than Adele."

Evidence:
"Cyndi Lauper won the Grammy Award for Best New Artist in 1985."

Atomic fact:
{
  "id": "f1",
  "subject": "Cyndi Lauper",
  "predicate": "won Best New Artist in",
  "object": "?year",
  "role": "anchor",
  "depends_on": []
}

Filled fact:
{
  "subject": "Cyndi Lauper",
  "predicate": "won Best New Artist in",
  "object": "?year"
}

Selected question:
When did Cyndi Lauper win the Best New Artist award?

Output:
{
  "fact_id": "f1",
  "question": "When did Cyndi Lauper win the Best New Artist award?",
  "answer": "1985",
  "status": "supported",
  "evidence_span": "Cyndi Lauper won the Grammy Award for Best New Artist in 1985.",
  "bindings_update": {
    "?year": "1985"
  }
}
"""

def get_answer_prompt(claim, evidence, atomic_fact, filled_fact, question_plan, bindings, question):
    prompt = answer_instruct + "\n\n"
    prompt += answer_examples + "\n\n"
    prompt += "Now solve the following.\n\n"
    prompt += f"Claim:\n\"{claim}\"\n\n"
    prompt += f"Evidence:\n\"{evidence}\"\n\n"
    prompt += f"Atomic fact:\n{json.dumps(atomic_fact, indent=2, ensure_ascii=False)}\n\n"
    prompt += f"Filled fact:\n{json.dumps(filled_fact, indent=2, ensure_ascii=False)}\n\n"
    prompt += f"Current bindings:\n{json.dumps(bindings, indent=2, ensure_ascii=False)}\n\n"
    prompt += f"Question templates:\n{json.dumps(question_plan, indent=2, ensure_ascii=False)}\n\n"
    prompt += f"Selected question:\n{question}\n\n"
    prompt += "Return JSON only."
    return prompt