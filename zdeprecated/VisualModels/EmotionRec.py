# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: Emotion recognition using gpt-4o

import aiohttp
import asyncio
import os
import time
import random
import base64
import requests
from PIL import Image
import io
from io import BytesIO
from Const import EMOTION_TO_ANIM
from Utils.OpenAIClient import api_key

def encode_image(image_path):
	with open(image_path, 'rb') as image_file:
		return base64.b64encode(image_file.read()).decode('utf-8')


class EmotionRecognizer:
    def __init__(self):
        self.headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {api_key}"}
        self.payload = {
		"model": "gpt-4o", # "gpt-4-vision-preview",
		"messages": [
					{
						"role": "user",
					}
				],
		"max_tokens": 250,}
    
    async def on_emotion_recog_for_vle(self, frame:bytes):
        img = Image.open(BytesIO(frame))
        b = io.BytesIO()
        img.save(b, 'png')
        base64_image = base64.b64encode(b.getvalue()).decode('utf-8')
       
        content = [
            {"type": "text", 
            "text": """You are talking with the person in front of you and guess the person's emotion base on the observation in the form of image, candidate_emotions are: -1.not provided,
             -1.other, 0.neutral, 1.surprise, 2.fear, 3.sadness, 4.joy, 5.disgust, 6.anger.
             you should provide the emotion only, e.g., 1.surprise.
             """},
            {"type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }},
        ]
        self.payload['messages'][0]['content'] = content
        async with aiohttp.ClientSession() as session:
            response = await session.post(url="https://api.openai.com/v1/chat/completions",
                                        headers=self.headers,
                                        json=self.payload)
            res = await response.json()
            msg = res['choices'][0]['message']['content']
            emo_label = int(msg.split('.')[0])
            emo_anim_lst = EMOTION_TO_ANIM.get(emo_label, [])
            print(f'emo list: {emo_anim_lst} \n****')
            if not emo_anim_lst:
                return ''
            return random.choice(emo_anim_lst)


    async def on_emotion_recog_task(self, frame:bytes):
        img = Image.open(BytesIO(frame))
        # img.save('debug.png')   # TODO 
        b = io.BytesIO()
        # print(f'type of bytes io: {type(b)} \n *******')
        img.save(b, 'png')
        base64_image = base64.b64encode(b.getvalue()).decode('utf-8')
        # base64_image = encode_image('/home/penny/pycharmprojects/AmecaBackend/Assets/img/image_phone.png')
        content = [
            {"type": "text", 
            "text": """You are talking with the person in front of you and guess the person's emotion base on the observation in the form of image, candidate_emotions are: -1.not provided,
             -1.other, 0.neutral, 1.surprise, 2.fear, 3.sadness, 4.joy, 5.disgust, 6.anger.
             you should provide the emotion first followed by some analysis, e.g.,
             1.happy
             you appears to be happy, i see your smile on your face.
             """},
            {"type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }},
        ]
        self.payload['messages'][0]['content'] = content
        async with aiohttp.ClientSession() as session:
            response = await session.post(url="https://api.openai.com/v1/chat/completions",
                                        headers=self.headers,
                                        json=self.payload)
            res = await response.json()
            # print(f'response:::: {res} \n*****')
            msg = res['choices'][0]['message']['content']
            print(msg)
            spts = msg.split('\n')
            analysis = ''.join(spts[1:])

            emo_label = spts[0]  # -1.not provided
            emo_label = int(emo_label.split('.')[0])
            emo_anim_lst = EMOTION_TO_ANIM.get(emo_label)
            if not emo_anim_lst:
                return '', analysis
            return random.choice(emo_anim_lst), analysis


# ===== example msg ======
''' -1.not provided

Based on the provided image, I cannot see your face or any significant facial features that would allow me to observe and identify your emotions.
type  spts: <class 'list'>, 
 ['-1.not provided', '', 'Based on the provided image, I cannot see your face or any significant facial features that would allow me to observe and identify your emotions.']
'''
# ==========================

emo_recognizer = EmotionRecognizer()


if __name__ == "__main__":
    emo_recog = EmotionRecognizer()
    asyncio.run(emo_recog.on_emotion_recog_task('b'))
    pass
