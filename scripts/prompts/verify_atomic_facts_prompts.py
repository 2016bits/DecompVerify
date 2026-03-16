import json

verify_instruct = """You are an expert fact-checking assistant.

Given:
1. a claim
2. an evidence passage
3. one filled atomic fact
4. the question used in the previous answering step
5. the answer produced in the previous answering step
6. the previous status and evidence span

Your task:
Determine whether this filled atomic fact is supported, contradicted, or insufficiently supported by the evidence.

Definitions:
- support: the evidence clearly supports the filled atomic fact
- contradict: the evidence clearly conflicts with the filled atomic fact
- insufficient: the evidence does not provide enough information to determine whether the fact is true or false

Requirements:
1. Use only the provided evidence.
2. Focus on the filled atomic fact, not the whole claim.
3. Return JSON only.
4. Keep the reason short and precise.
5. Copy a short supporting or conflicting evidence span when possible.
6. Do not infer beyond the evidence.
7. If the evidence is partial or ambiguous, use "insufficient".

Output schema:
{
  "fact_id": "f1",
  "verification_label": "support|contradict|insufficient",
  "reason": "...",
  "evidence_span": "..."
}
"""

verify_examples = """Examples:

Example 1:

Claim:
"Jack McFarland is the best known role of the host of the 64th Annual Tony Awards."

Evidence:
"Sean Patrick Hayes is an American actor. He is best known for his role as Jack McFarland on the NBC sitcom Will & Grace. The host of the 64th Annual Tony Awards was Sean Hayes."

Filled atomic fact:
{
  "id": "f1",
  "subject": "Sean Hayes",
  "predicate": "played",
  "object": "Jack McFarland",
  "role": "bridge",
  "depends_on": []
}

Question used:
Who played Jack McFarland?

Answer from previous step:
Sean Hayes

Previous status:
supported

Previous evidence span:
He is best known for his role as Jack McFarland on the NBC sitcom Will & Grace.

Output:
{
  "fact_id": "f1",
  "verification_label": "support",
  "reason": "The evidence states that Sean Hayes is known for his role as Jack McFarland.",
  "evidence_span": "He is best known for his role as Jack McFarland on the NBC sitcom Will & Grace."
}


Example 2:

Claim:
"The director of the film starring Tom Hanks was born in California."

Evidence:
"Saving Private Ryan starred Tom Hanks. The film was directed by Steven Spielberg. Steven Spielberg was born in Cincinnati, Ohio."

Filled atomic fact:
{
  "id": "f3",
  "subject": "Steven Spielberg",
  "predicate": "born in",
  "object": "California",
  "role": "verify",
  "depends_on": ["f2"]
}

Question used:
Was Steven Spielberg born in California?

Answer from previous step:
No

Previous status:
contradicted

Previous evidence span:
Steven Spielberg was born in Cincinnati, Ohio.

Output:
{
  "fact_id": "f3",
  "verification_label": "contradict",
  "reason": "The evidence gives a different birthplace for Steven Spielberg.",
  "evidence_span": "Steven Spielberg was born in Cincinnati, Ohio."
}


Example 3:

Claim:
"Cyndi Lauper won the Best New Artist award earlier than Adele."

Evidence:
"Cyndi Lauper won the Grammy Award for Best New Artist in 1985."

Filled atomic fact:
{
  "id": "f2",
  "subject": "Adele",
  "predicate": "won Best New Artist in",
  "object": "2012",
  "role": "anchor",
  "depends_on": []
}

Question used:
When did Adele win the Best New Artist award?

Answer from previous step:


Previous status:
insufficient

Previous evidence span:


Output:
{
  "fact_id": "f2",
  "verification_label": "insufficient",
  "reason": "The evidence does not mention Adele or the year she won the award.",
  "evidence_span": ""
}


Example 4:

Claim:
"Puerto Rico is not an unincorporated territory of the United States."

Evidence:
"Puerto Rico is an unincorporated territory of the United States."

Filled atomic fact:
{
  "id": "f1",
  "subject": "Puerto Rico",
  "predicate": "is",
  "object": "an unincorporated territory of the United States",
  "role": "verify",
  "depends_on": []
}

Question used:
Is it true that Puerto Rico is an unincorporated territory of the United States?

Answer from previous step:
Yes

Previous status:
supported

Previous evidence span:
Puerto Rico is an unincorporated territory of the United States.

Output:
{
  "fact_id": "f1",
  "verification_label": "support",
  "reason": "The evidence explicitly states that Puerto Rico is an unincorporated territory of the United States.",
  "evidence_span": "Puerto Rico is an unincorporated territory of the United States."
}
"""


def get_verify_prompt(claim, evidence, answer_item):
    prompt = verify_instruct + "\n\n"
    prompt += verify_examples + "\n\n"
    prompt += "Now solve the following.\n\n"
    prompt += f"Claim:\n\"{claim}\"\n\n"
    prompt += f"Evidence:\n\"{evidence}\"\n\n"
    prompt += f"Filled atomic fact:\n{json.dumps(answer_item['filled_fact'], indent=2, ensure_ascii=False)}\n\n"
    prompt += f"Question used:\n{answer_item['question_used']}\n\n"
    prompt += f"Answer from previous step:\n{answer_item['answer']}\n\n"
    prompt += f"Previous status:\n{answer_item['status']}\n\n"
    prompt += f"Previous evidence span:\n{answer_item['evidence_span']}\n\n"
    prompt += "Return JSON only."
    return prompt
