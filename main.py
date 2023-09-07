import pandas as pd
import json
import numpy as np
import gradio as gr
import random
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

# set delimiter
delimiter = "####"

# set train_data for text-to-image generate
train_data = ""
for item in example:
    train_data += item
    train_data += delimiter

# global variable
last_content = ""
titles = []
publishes = []
record = []


def btn_1(chat_history):
    return button_function(chat_history, btn=1)


def btn_2(chat_history):
    return button_function(chat_history, btn=2)


def btn_3(chat_history):
    return button_function(chat_history, btn=3)


def btn_next(chat_history):
    global publishes
    global record
    global last_content

    publishes.append(last_content)
    last_content = ""

    if len(publishes) == 7:
        chat_history.append(("故事结束啦！", "请点击'Publish'按钮发布故事！"))
    else:
        format_1 = """
        {"Brief Introduction" : ["Introduction1", "Introduction2", "Introduction3"]}
        """
        prompt = f"请生成Page{len(publishes)}的故事梗概，参考格式{format_1}"
        record.append({"role": "user", "content": prompt})

        response = Azure_OpenAI_API.get_completion_from_messages(messages=record)
        record.append({"role": "assistant", "content": response})
        data = json.loads(response)

        output = f"""
        请选择第{len(publishes)}页的故事：
        1. {data["Brief Introduction"][0]}
        2. {data["Brief Introduction"][1]}
        3. {data["Brief Introduction"][2]}
        """
        chat_history.append((f"下一页！", output))

    return chat_history


def btn_refresh(chat_history):
    global record
    global last_content

    prompt = f"我对结果不满意，请按照格式重新生成"
    record.append({"role": "user", "content": prompt})

    response = Azure_OpenAI_API.get_completion_from_messages(messages=record)
    record.append({"role": "assistant", "content": response})
    data = json.loads(response)

    if last_content == "":
        output = f"""
        请选择第{len(publishes)}页的故事：
        1. {data["Brief Introduction"][0]}
        2. {data["Brief Introduction"][1]}
        3. {data["Brief Introduction"][2]}
        """
    else:
        last_content = data["Story"]
        output = f"""
        第{len(publishes)}页：
        {last_content}
        """

    chat_history.append(("刷新！", output))
    return chat_history


def btn_start(chat_history):
    global titles
    global publishes
    publishes = []

    system_prompt = """
    你是一位儿童文学作家，你的读者主要是小学和学龄前儿童。
    我们正准备创作一个儿童故事，请你提出与输入的关键词有关的3个故事标题。
    要求：
    1. 请使用JSON格式组织你的回答。
    2. 每一个标题不超过10个中文字符
    """

    input_example = "大海"

    output_example = """
    {"titles": ["小鱼的冒险", "海底奇遇记", "海洋王国的守护者"]}
    """

    topic = random.choice(["大海", "农场", "冒险", "动物", "森林"])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_example},
        {"role": "assistant", "content": output_example},
        {"role": "user", "content": topic},
    ]

    response = Azure_OpenAI_API.get_completion_from_messages(messages=messages)
    data = json.loads(response)
    titles = [data["titles"][0], data["titles"][1], data["titles"][2]]

    output = f"""
    请选择你喜欢的主题：
    1. {data["titles"][0]}
    2. {data["titles"][1]}
    3. {data["titles"][2]}
    """

    chat_history.append(("请为我讲个故事吧！", output))
    return chat_history


def btn_publish():
    global publishes
    text = ""
    for i in range(len(publishes)):
        if i > 0:
            text += f"第{i}页： "
        text += publishes[i]
        if i < 7:
            text += "\n"
    return text


def button_function(chat_history, btn=1):
    global record
    global publishes

    format_1 = """
    {"Brief Introduction" : ["Introduction1", "Introduction2", "Introduction3"]}
    """
    format_2 = """
    {"Story" : "Content"}
    """

    if len(publishes) == 0:
        global titles
        title = titles[btn-1]
        publishes.append(title)

        system_prompt = f"""
        你是一位儿童文学作家，你的读者主要是小学和学龄前儿童。
        你正在和儿童共同创作一个儿童故事，故事的题目是{title}。
        整个格式将由六页组成，请注意剧情进程的控制以保证故事的完整。
        在创作过程中，你需要先为每一页提供三个故事梗概，待儿童选择后再进行详细的描写。
        """

        prompt = f"请生成Page1的故事梗概，参考格式{format_1}"

        record = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = Azure_OpenAI_API.get_completion_from_messages(messages=record)
        record.append({"role": "assistant", "content": response})
        data = json.loads(response)

        output = f"""
        请选择第1页的故事：
        1. {data["Brief Introduction"][0]}
        2. {data["Brief Introduction"][1]}
        3. {data["Brief Introduction"][2]}
        """
        chat_history.append((f"我选择第{btn}个题目！", output))
    else:
        prompt = f"""
        我选择第{btn}个，请在梗概的基础上补充完成本页的故事细节，请注重场景描写和角色刻画
        要求：
        1. 不超过30个中文字符 
        2. 参考格式{format_2}
        """
        record.append({"role": "user", "content": prompt})

        response = Azure_OpenAI_API.get_completion_from_messages(messages=record)
        record.append({"role": "assistant", "content": response})
        data = json.loads(response)
        global last_content
        last_content = data["Story"]

        output = f"""
        第{len(publishes)}页：
        {last_content}
        """
        chat_history.append((f"我选择第{btn}个！", output))

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
    with gr.Accordion(label="Ink Story", open=True):
        gr.Markdown("## Ink Diffusion--StoryTeller")
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(label="Chatbot",
                                         show_label=True,
                                         height=320)
                with gr.Column(scale=1):
                    button_start = gr.Button("Start")
                    button_1 = gr.Button("1")
                    button_2 = gr.Button("2")
                    button_3 = gr.Button("3")
                    button_refresh = gr.Button("Refresh")
                    button_next = gr.Button("Next")
            with gr.Column():
                button_publish = gr.Button("Publish")
                publish_box = gr.Textbox(label="Ink Publish", show_copy_button=True)
                button_clear = gr.ClearButton(components=[chatbot, publish_box], value="Clear")

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
    button_next.click(fn=btn_next, inputs=[chatbot], outputs=[chatbot])
    button_refresh.click(fn=btn_refresh, inputs=[chatbot], outputs=[chatbot])
    button_start.click(fn=btn_start, inputs=[chatbot], outputs=[chatbot])
    button_publish.click(fn=btn_publish, inputs=[], outputs=[publish_box])
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
