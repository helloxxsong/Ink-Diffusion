import os
import openai

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_type = "azure"
openai.api_key = os.environ['AZURE_OPENAI_API_KEY']
openai.api_base = os.environ['AZURE_OPENAI_API_BASE']
openai.api_version = "2023-05-15"


def get_completion(prompt, temperature=1.0):
    response = openai.ChatCompletion.create(
        deployment_id="Ink-Diffusion",
        model="gpt-35-turbo-16k",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content


def get_completion_from_messages(messages, temperature=1.0):
    response = openai.ChatCompletion.create(
        deployment_id="gpt-35-turbo-16k",
        model="gpt-35-turbo-16k",   
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content


print(get_completion(prompt="Hi"))
