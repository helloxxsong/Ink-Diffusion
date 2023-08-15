import openai
import pandas as pd
import numpy as np
import gradio as gr
import os
from dotenv import load_dotenv, find_dotenv

# read OPENAI_API_KEY from local .env file
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# get the directory path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# build a relative path to TTL_train_data.csv
csv_file_path = os.path.join(current_dir, 'TTL_train_data.csv')

# read examples from IIT_train_data.csv
IIT_train_data = pd.read_csv(csv_file_path)
example = np.array(IIT_train_data['prompt'])

# set delimiter
delimiter = "####"

# set train_data for text-to-image generate
train_data = ""
for item in example:
    train_data += item
    train_data += delimiter


def button_function(level):
    return level


# create interactive interface with gradio
with gr.Blocks() as demo:
    gr.Markdown("# Ink Diffusion")
    with gr.Row():
        with gr.Column(scale=4):
            outputTextbox_1 = gr.Textbox(label="Output", lines=5)
        with gr.Column(scale=1):
            button_1 = gr.Button("1")
            button_2 = gr.Button("2")
            button_3 = gr.Button("3")
            button_4 = gr.Button("4")
            button_refresh = gr.Button("refresh")
    level = gr.Slider(label="Level", maximum=10, minimum=1, step=1, value=5)

    button_1.click(fn=button_function, inputs=[level], outputs=[outputTextbox_1])

gr.close_all()
demo.launch()
