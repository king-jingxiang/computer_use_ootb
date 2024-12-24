import ast
import base64
import os
from io import BytesIO

import requests
from PIL import Image, ImageDraw
from openai import OpenAI
from qwen_vl_utils import process_vision_info


class ShowUIModel:
    def __init__(self, model_name, detail="auto", system_prompt=None, system_message=None, base_url=None, api_key=None,
                 min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28):
        # detail: auto, low, high
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model_name
        self.detail = detail
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = "You are a helpful assistant."
        if system_message:
            self.system_message = system_message
        else:
            # self.system_message = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
            self.system_message = """You are an assistant trained to navigate the desktop screen. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
1. CLICK: Click on an element, value is not applicable and the position [x,y] is required. 
2. INPUT: Type a string into an element, value is a string to type and the position [x,y] is required. 
3. HOVER: Hover on an element, value is not applicable and the position [x,y] is required.
4. ENTER: Enter operation, value and position are not applicable.
5. SCROLL: Scroll the screen, value is the direction to scroll and the position is not applicable.
6. ESC: ESCAPE operation, value and position are not applicable.
7. PRESS: Long click on an element, value is not applicable and the position [x,y] is required. """

    def draw_point(self, image_input, point=None, radius=5):
        if isinstance(image_input, str):
            image = Image.open(BytesIO(requests.get(image_input).content)) if image_input.startswith(
                'http') else Image.open(image_input)
        else:
            image = image_input

        image = self.process_vision_info(image)

        if point:
            if isinstance(point, str):
                point = ast.literal_eval(point)
            x, y = point[0] * image.width, point[1] * image.height
            ImageDraw.Draw(image).ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
        return image

    # TODO 图片预先进行像素缩放处理, 像素点坐标后处理
    def process_image(self, image_path, query):
        image = Image.open(image_path)
        processed_image = self.process_vision_info(image)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.system_message,
                        },
                        {
                            "type": "text",
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{processed_image}"
                            }
                        }
                    ]
                }
            ]
        )
        return completion.choices[0].message.content

    def process_message(self, messages):
        messages_copy = messages.copy()
        for msg in messages_copy:
            if msg["role"] == "user":
                for i, content in enumerate(msg["content"]):
                    if content["type"] == "image":
                        processed_image = self.process_vision_info(content["image"])
                        msg["content"][i] = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{processed_image}"
                            }
                        }

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages_copy,
        )
        return completion.choices[0].message.content

    def process_vision_info(self, image):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                ]
            }
        ]
        image_inputs, _ = process_vision_info(messages)
        buffered = BytesIO()
        image_inputs[0].save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return img_base64

# if __name__ == '__main__':
#     model = ShowUIModel(model_name="/models/AI-ModelScope/ShowUI-2B", base_url="http://10.1.30.3:48001/v1",
#                         api_key="123456")
#     print(model.process_image("/Users/charles/Downloads/dog_and_girl.jpeg", "dog."))
