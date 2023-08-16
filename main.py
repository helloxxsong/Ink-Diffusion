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

# global variable
record = []


def btn_1(chat_history):
    return button_function(chat_history, btn=1)


def btn_2(chat_history):
    return button_function(chat_history, btn=2)


def btn_3(chat_history):
    return button_function(chat_history, btn=3)


def btn_4(chat_history):
    return button_function(chat_history, btn=4)


def btn_refresh(chat_history):
    return button_function(chat_history, btn=0)


def btn_start(chat_history):
    global record
    chat_history.append(("hi", "hello"))
    return chat_history


def btn_publish():
    global record
    text = ""
    for item in record:
        text += item
        text += "\n"
    return text


def button_function(chat_history, btn=0):
    global record
    message = str(btn)
    bot_message = "hello" + message
    record.append(bot_message)
    chat_history.append((message, bot_message))
    return chat_history


def generate_function(user_input, level):
    return "", user_input, level


# create interactive interface with gradio
with (gr.Blocks() as demo):
    gr.Markdown("# Ink Diffusion")
    with gr.Accordion(label="Function 1", open=True):
        gr.Markdown("## Ink Diffusion")
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(label="Function 1",
                                         show_label=True,
                                         height=275)
                with gr.Column(scale=1):
                    button_1 = gr.Button("1")
                    button_2 = gr.Button("2")
                    button_3 = gr.Button("3")
                    button_4 = gr.Button("4")
                    button_refresh = gr.Button("Refresh")
            with gr.Row():
                button_start = gr.Button("Start")
                button_publish = gr.Button("Publish")
            with gr.Column():
                publish = gr.Textbox(label="Ink Publish", show_copy_button=True)
                button_clear = gr.ClearButton(components=[chatbot, publish], value="Clear")

    with gr.Accordion(label="Function 2", open=False):
        gr.Markdown("## Ink Diffusion")
        with gr.Column():
            story_input = gr.Textbox(label="Paste story here")
            button_run = gr.Button("Generate")
            with gr.Row():
                image_show = gr.Image()
                with gr.Column():
                    label = gr.Textbox(label="Story")
                    audio = gr.Microphone(type="filepath", format="mp3", show_edit_button=False)
                    with gr.Row():
                        image_regenerate = gr.Button("Regenerate")
                        button_next = gr.Button("Next")
            video = gr.Video()

    with gr.Accordion(label="Function 3", open=False):
        gr.Markdown("## Ink Diffusion")
        with gr.Column():
            user_input = gr.Textbox(label="Input")
            with gr.Row():
                level = gr.Slider(label="Level", maximum=10, minimum=1, step=1, value=5)
                tmp = gr.Slider(label="Temperature", maximum=2.0, minimum=0.0, step=0.1, value=1.0)
            button_generate = gr.Button("Generate")
            bot_output = gr.Textbox(label="Output", show_copy_button=True)
            bot_prompt = gr.Textbox(label="Output", show_copy_button=True)
            image_output = gr.Image(label="Result")

    button_1.click(fn=btn_1, inputs=[chatbot], outputs=[chatbot])
    button_2.click(fn=btn_2, inputs=[chatbot], outputs=[chatbot])
    button_3.click(fn=btn_3, inputs=[chatbot], outputs=[chatbot])
    button_4.click(fn=btn_4, inputs=[chatbot], outputs=[chatbot])
    button_refresh.click(fn=btn_refresh, inputs=[chatbot], outputs=[chatbot])
    button_start.click(fn=btn_start, inputs=[chatbot], outputs=[chatbot])
    button_publish.click(fn=btn_publish, inputs=[], outputs=[publish])
    button_generate.click(fn=generate_function,
                          inputs=[user_input, level],
                          outputs=[user_input, bot_output, bot_prompt])

gr.close_all()
demo.launch()
