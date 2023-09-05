import openai
import os
import requests
from PIL import Image
import time

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
openai.api_type = "azure"
openai.api_key = os.environ['AZURE_OPENAI_API_KEY']
openai.api_base = os.environ['AZURE_OPENAI_API_BASE']
openai.api_version = '2023-06-01-preview'


def get_image(prompt):
    # Set the directory for the stored image
    image_dir = os.path.join(os.curdir, 'images')

    # If the directory doesn't exist, create it
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    # Initialize the image path (note the filetype should be png)
    current_time = time.strftime("%d%H%M%S")
    image_path = os.path.join(image_dir, f'generated_image_{current_time}.png')

    # Retrieve the generated image
    response = openai.Image.create(
        prompt=prompt,
        size='512x512',
        n=4
    )
    image_url = response["data"][0]["url"]  # extract image URL from response
    generated_image = requests.get(image_url).content  # download the image
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)

    # Display the image in the default image viewer
    image = Image.open(image_path)
    image.show()
