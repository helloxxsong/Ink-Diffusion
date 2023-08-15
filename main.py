import openai
import pandas as pd
import numpy as np
import json
import gradio as gr
import os
from dotenv import load_dotenv, find_dotenv

# read OPENAI_API_KEY from local .env file
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# set delimiter
delimiter = "####"

# read examples from IIT_train_data.csv
IIT_train_data = pd.read_csv('./TTI_train_data.csv')
example = np.array(IIT_train_data['prompt'])

# set train_data for image generate
train_data = "1"
for item in example:
    train_data += item
    train_data += delimiter

print(train_data)
