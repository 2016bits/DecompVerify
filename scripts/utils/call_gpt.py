import os
from openai import OpenAI

endpoint = "https://open1027.openai.azure.com/openai/v1/"
deployment_name = "gpt-4o"
api_key = os.getenv("AZURE_OPENAI_API_KEY")  

client = OpenAI(    
    base_url=endpoint,
    api_key=api_key
)

def llm(prompt, stop=["\n"]):
    completion = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "user",
             "content": prompt}
        ]
    )
    return completion.choices[0].message.content

print(llm("What is the capital of France?"))