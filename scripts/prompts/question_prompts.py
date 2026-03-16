import json

question_instruct = """You are an expert planner for multi-hop fact verification.

Given a claim and one atomic fact from its decomposition, generate question templates for verifying this atomic fact.

Requirements:
1. Return JSON only.
2. Be faithful to the atomic fact.
3. If role is "bridge", generate questions that identify the unknown variable.
4. If role is "verify", generate questions that verify whether the relation holds.
5. If role is "anchor", generate questions that retrieve the missing attribute value.
6. If a variable is unresolved, keep it as a placeholder such as ?manager, ?film, ?director, ?t1.
7. Keep questions natural and concise.
8. Prefer question forms that are easy to answer directly from evidence.
9. The output must follow the schema exactly.

Output schema:
{{
  "fact_id": "f1",
  "role": "bridge|verify|anchor",
  "question_templates": {{
    "primary": "...",
    "alternate": "...",
    "boolean": "...",
    "boolean_template": "..."
  }}
}}
"""

question_examples = """Examples:

Claim:
"The director of the film starring Tom Hanks was born in California."

Atomic fact:
{
  "id": "f1",
  "subject": "Tom Hanks",
  "predicate": "starred in",
  "object": "?film",
  "role": "bridge",
  "depends_on": []
}

Output:
{
  "fact_id": "f1",
  "role": "bridge",
  "question_templates": {
    "primary": "What film did Tom Hanks star in?",
    "alternate": "Which film starred Tom Hanks?",
    "boolean": "",
    "boolean_template": "Did Tom Hanks star in {value}?"
  }
}

Claim:
"The director of the film starring Tom Hanks was born in California."

Atomic fact:
{
  "id": "f2",
  "subject": "?film",
  "predicate": "directed by",
  "object": "?director",
  "role": "bridge",
  "depends_on": ["f1"]
}

Output:
{
  "fact_id": "f2",
  "role": "bridge",
  "question_templates": {
    "primary": "Who directed ?film?",
    "alternate": "Who was the director of ?film?",
    "boolean": "",
    "boolean_template": "Did {value} direct ?film?"
  }
}

Claim:
"The director of the film starring Tom Hanks was born in California."

Atomic fact:
{
  "id": "f3",
  "subject": "?director",
  "predicate": "born in",
  "object": "California",
  "role": "verify",
  "depends_on": ["f2"]
}

Output:
{
  "fact_id": "f3",
  "role": "verify",
  "question_templates": {
    "primary": "Where was ?director born?",
    "alternate": "Was ?director born in California?",
    "boolean": "Is it true that ?director was born in California?",
    "boolean_template": ""
  }
}

Claim:
"Cyndi Lauper won the Best New Artist award earlier than Adele."

Atomic fact:
{
  "id": "f1",
  "subject": "Cyndi Lauper",
  "predicate": "won Best New Artist in",
  "object": "?t1",
  "role": "anchor",
  "depends_on": []
}

Output:
{
  "fact_id": "f1",
  "role": "anchor",
  "question_templates": {
    "primary": "When did Cyndi Lauper win the Best New Artist award?",
    "alternate": "In what year did Cyndi Lauper win the Best New Artist award?",
    "boolean": "",
    "boolean_template": ""
  }
}

Claim:
"Jack McFarland is the best known role of the host of the 64th Annual Tony Awards."

Atomic fact:
{
  "id": "f1",
  "subject": "?actor",
  "predicate": "played",
  "object": "Jack McFarland",
  "role": "bridge",
  "depends_on": []
}

Output:
{
  "fact_id": "f1",
  "role": "bridge",
  "question_templates": {
    "primary": "Who played Jack McFarland?",
    "alternate": "Which actor played Jack McFarland?",
    "boolean": "",
    "boolean_template": "Did {value} play Jack McFarland?"
  }
}

Claim:
"Jack McFarland is the best known role of the host of the 64th Annual Tony Awards."

Atomic fact:
{
  "id": "f2",
  "subject": "?actor",
  "predicate": "hosted",
  "object": "64th Annual Tony Awards",
  "role": "bridge",
  "depends_on": ["f1"]
}

Output:
{
  "fact_id": "f2",
  "role": "bridge",
  "question_templates": {
    "primary": "Who hosted the 64th Annual Tony Awards?",
    "alternate": "Who was the host of the 64th Annual Tony Awards?",
    "boolean": "",
    "boolean_template": "Did {value} host the 64th Annual Tony Awards?"
  }
}

Claim:
"Puerto Rico is not an unincorporated territory of the United States."

Atomic fact:
{
  "id": "f1",
  "subject": "Puerto Rico",
  "predicate": "is",
  "object": "an unincorporated territory of the United States",
  "role": "verify",
  "depends_on": []
}

Output:
{
  "fact_id": "f1",
  "role": "verify",
  "question_templates": {
    "primary": "What is Puerto Rico's political status with respect to the United States?",
    "alternate": "Is Puerto Rico an unincorporated territory of the United States?",
    "boolean": "Is it true that Puerto Rico is an unincorporated territory of the United States?",
    "boolean_template": ""
  }
}
"""

def get_question_prompt(claim, atomic_fact):
    prompt = question_instruct + question_examples + f'\nClaim: "{claim}"\nAtomic fact: {json.dumps(atomic_fact)}\nOutput:'
    return prompt
