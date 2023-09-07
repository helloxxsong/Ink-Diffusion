import pandas as pd
import json
import numpy as np
import gradio as gr
import os

import Azure_DALLE2
import Azure_OpenAI_API

# get the directory path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# build a relative path to TTL_train_data.csv
csv_file_path = os.path.join(current_dir, 'TTL_train_data.csv')

# read examples from IIT_train_data.csv
IIT_train_data = pd.read_csv(csv_file_path)
example = np.array(IIT_train_data['prompt'])
print(len(example))

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


def generate_function_1(user_input, level, tmp):
    if user_input == '':
        return "Please enter the content."

    system_prompt = f"""
    你是一个儿童文学作家，致力于帮助儿童去创作图画书。
    你的任务是创作一些积极的、富有表现力的场景描写或角色特写。
    你创作的内容必须基于用户输入的内容。
    你可以为描写增加细节使画面更加生动、更具吸引力。
    出于教育目的，你的需要辅助创作十个不同等级的回答：从原始的输入开始，回答的内容逐渐丰富。
    要求：
    1. 请使用JSON格式组织你的回答。
    2. 请注重更具视觉表现力的场景描写以及角色外貌的刻画，弱化心理描写等不可视内容。
    3. 任何等级的回答请不要超出50个中文字符。
    """

    input_example = "一个女孩站在山顶上"

    output_example = """
    {
    "Level0" : "一个女孩站在山顶上",
    "Level1" : "在一个高高的山顶上，一个可爱的女孩站在那里",
    "Level2" : "在这个壮丽的山顶上，一个拥有长长金色头发的女孩子站在那里",
    "Level3" : "在这座巍峨山峰的顶端，一位年轻的女孩站在那里，她的头发像金色的瀑布一样垂落在她的肩膀上",
    "Level4" : "在这座壮丽山峰的巅峰处，一个娇小的女孩矗立着。她的金色长发在微风中飘动，如同太阳的光芒洒在大地上",
    "Level5" : "在这个被云雾环绕的山顶上，一个优雅的女孩站在那里。她的蓝色连衣裙在微风中轻轻飘动，与她闪亮的眼睛相映成趣",
    "Level6" : "在这座神秘山峰的巅峰处，一个妖娆的女孩屹立不倒。她的金色长发犹如黄金瀑布，从她的头顶上垂落，悠然自得",
    "Level7" : "在这座壮丽山峰的巅峰上，一个美丽的女孩站在那里。她的头发犹如细细的丝线一样柔顺，闪烁着微弱的光芒",
    "Level8" : "在这个宏伟壮丽的山峰之巅，一个娇柔的女孩屹立不倒。她的金色长发如同烈火一般燃烧着，映衬出她明亮的眼睛",
    "Level9" : "在这座壮丽山峰的巅峰上，一个飒爽的女孩站在那里。她的头发犹如黄金海洋般波浪起伏，眼睛里闪烁着智慧的光芒，仿佛能看透一切"
    }
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_example},
        {"role": "assistant", "content": output_example},
        {"role": "user", "content": user_input},
    ]

    try:
        response = Azure_OpenAI_API.get_completion_from_messages(messages=messages, temperature=float(tmp))
        data = json.loads(response)
        return data[f"Level{int(level)}"]
    except Exception as e:
        return e


def generate_function_2(bot_output):
    if bot_output == '':
        return "Please enter the content."

    system_prompt = f"""
    You are a Text-To-Image prompt generator.
    Your task is to read the content of scene I give and generator prompts that can be used in DALL-E to generator images for children's picture book.
    You can add some descriptions in the content to make it more vivid, but do not change the meaning of the content.
    Given the example prompts delimited by {delimiter}：
    <examples>
    {train_data}
    </examples>
    You can imitate the styles of examples.
    Please return in English and format your response as a JSON object with "Prompt" as the key.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": bot_output},
    ]

    try:
        response = Azure_OpenAI_API.get_completion_from_messages(messages=messages)
        data = json.loads(response)
        return data["Prompt"]
    except Exception as e:
        return e


def generate_function_3(bot_prompt):
    return Azure_DALLE2.get_image(prompt=bot_prompt)


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

    with gr.Accordion(label="ToolBox", open=False):
        gr.Markdown("## Ink Diffusion--ToolBox")
        with gr.Column():
            user_input = gr.Textbox(label="Input")
            with gr.Row():
                level = gr.Slider(label="Level", maximum=9, minimum=1, step=1, value=5)
                tmp = gr.Slider(label="Temperature", maximum=1.0, minimum=0.0, step=0.1, value=1.0)
            button_generate_1 = gr.Button("GPT Assistant")
            bot_output = gr.Textbox(label="Output", show_copy_button=True)
            button_generate_2 = gr.Button("Generate Prompt")
            bot_prompt = gr.Textbox(label="Prompt", show_copy_button=True)
            button_generate_3 = gr.Button("Generate Image")
            image_output = gr.Image(label="Result")

    button_1.click(fn=btn_1, inputs=[chatbot], outputs=[chatbot])
    button_2.click(fn=btn_2, inputs=[chatbot], outputs=[chatbot])
    button_3.click(fn=btn_3, inputs=[chatbot], outputs=[chatbot])
    button_4.click(fn=btn_4, inputs=[chatbot], outputs=[chatbot])
    button_refresh.click(fn=btn_refresh, inputs=[chatbot], outputs=[chatbot])
    button_start.click(fn=btn_start, inputs=[chatbot], outputs=[chatbot])
    button_publish.click(fn=btn_publish, inputs=[], outputs=[publish])
    button_generate_1.click(fn=generate_function_1,
                            inputs=[user_input, level, tmp],
                            outputs=[bot_output])
    button_generate_2.click(fn=generate_function_2,
                            inputs=[bot_output],
                            outputs=[bot_prompt])
    button_generate_3.click(fn=generate_function_3,
                            inputs=[bot_prompt],
                            outputs=[image_output])

gr.close_all()
demo.launch()
