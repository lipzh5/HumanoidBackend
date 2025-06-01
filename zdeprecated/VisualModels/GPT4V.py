# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

import aiohttp
import asyncio
import os
import time
import base64
import requests
from PIL import Image
import io
from io import BytesIO
from Utils.OpenAIClient import api_key

# def encode_image(image_path):
# 	with open(image_path, 'rb') as image_file:
# 		return base64.b64encode(image_file.read()).decode('utf-8')


headers = {
		"Content-Type": "application/json",
		"Authorization": f"Bearer {api_key}"
	}

payload = {
		"model": "gpt-4o", # "gpt-4-vision-preview",
		"messages": [
					{
						"role": "user",
					},
					{
						"role": "system",
						"content": [{"type": "text",
						"text": "You are an intelligent AI assistant, always ready to polish the answer to the visual-questions"
						}],
					}
				],
		"max_tokens": 300
}

async def save_img_from_bytes(img_bytes, output_dir, filename='vqa_img.jpg'):
	"""save image from byte stream to a file"""
	if not os.path.exists(output_dir):
		os.mkdirs(output_dir)
	with open(os.path.join(output_dir, filename), 'wb') as f:
		f.write(img_bytes)


def generate_img_url_from_bytes(img_bytes, ourput_dir, port=8000):
	"""save image from byte streams to files and generate url for it"""
	filename = f'image.jpg'
	save_img_from_bytes(img_bytes, output_dir, filename)
	return f'http://localhost:{port}/{filename}'

	# img_urls = []
	# for i, img_bytes in enumerate(img_bytes_list):
	# 	filename = f'image_{i}.jpg'
	# 	save_img_from_bytes(img_bytes, output_dir, filename)
	# 	img_urls.append(f'http://localhost:{port}/{filename}')
	# return img_urls


async def run_vqa_from_client_query(img_bytes, query:bytes):  # img = Image.frombytes('RGB', (1280, 720), img_bytes)
	if not api_key:
		return
	# print(f'type of image bytes: {type(img_bytes)}, len image bytes: {len(img_bytes)} \n ********')
	query = query.decode('utf-8')
	# img_url = generate_img_urls_from_bytes(img_bytes)  # for single frame
	# ====base64 encoded image====
	img = Image.open(BytesIO(img_bytes))
	b = io.BytesIO()
	# print(f'type of bytes io: {type(b)} \n *******')
	img.save(b, 'png')
	base64_image = base64.b64encode(b.getvalue()).decode('utf-8')
	# =====================================
	print(f'run vqa after encode image: {time.strftime("%X")}')
	# update query and image
	content = [
		{
			"type": "text",
			"text": query,
		},
		{
			"type": "image_url",
			"image_url": {
				# 'url': "https://10.6.37.210:8000/image_phone.png"
				"url": f"data:image/jpeg;base64,{base64_image}"
			}
		}
	]
	payload['messages'][0]['content'] = content
	print(f'run vqa starts send request: {time.strftime("%X")}')
	async with aiohttp.ClientSession() as session:
		response = await session.post(url="https://api.openai.com/v1/chat/completions",
									  headers=headers,
									  json=payload)
		res = await response.json()
		# print(res)
		# print(res['choices'][0])
		# print('type of message: ', type(res['choices'][0]['message']))
		msg = res['choices'][0]['message']['content']
		print(msg)
		print(f'run vqa get answer: {time.strftime("%X")}')
		return msg


if __name__ == "__main__":
	from Utils.OpenAIClient import client
	response = client.chat.completions.create(
		model='gpt-4o',
		messages=[
			{
				"role": "user",
				"content": [
					{"type": "text",  "text": "what is this in this image?"},
					{"type": "image_url",
						"image_url": {
						"url": "http://137.111.13.6:8000/image_phone.png"  # 137.111.13.6, 10.6.37.210
					},
					}
				]
			}
		],
		max_tokens=250,
	)

	# asyncio.run(video_captioning('../saved_frames_rollout/batch3'))
	# query = 'what is this in my hand?'
	# img_path = os.path.dirname(os.path.abspath(__file__))
	# img_path = os.path.join(img_path, '../assets/image_hair_dryer.png')
	# print(img_path)
	# print(f'os path exists? {os.path.exists(img_path)}')
	# print(time.strftime('%X'))
	# asyncio.run(run_vqa_from_client_query(None, b'what is this in the image?'))
	# print(time.strftime('%X'))

	pass




