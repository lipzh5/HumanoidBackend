# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: action generation using GPT
import os
import asyncio
from openai import AsyncOpenAI
from VisualModels.Hiera import video_recognizer
from CONF import gpt_model_name, base_prompt

file_path = os.path.dirname(os.path.abspath(__file__))
openai_api_key = ''
try:
	with open(os.path.join(file_path, "openai_key.txt")) as fd:
		openai_api_key = fd.read().strip()
		print('open ai key: ', openai_api_key)
except FileNotFoundError:
	print('could not find open ai key file')

client = AsyncOpenAI(api_key=openai_api_key)

LLM_CACHE = {}


async def gpt_call(engine, prompt, max_token=256, temperature=0, logprobs=1):
	_id = tuple((engine, prompt, max_token, temperature, logprobs))
	resp = LLM_CACHE.get(_id)
	if resp is None:
		resp = await client.completions.create(model=engine,
										 prompt=prompt,
										 temperature=temperature,
										 logprobs=logprobs)
		LLM_CACHE[_id] = resp
	print('response!!!!', resp)
	return resp


def post_process(response):  # Completion object
	choice = response.choices[0]
	text = choice.text.replace('[', '*').replace(']', '*').split('*')
	print('text: ', text[1])
	poses = text[1].split(',')
	poses = [pose.lstrip().rstrip() for pose in poses]
	return poses


async def on_pose_generation(human_action):
	# human_action = video_recognizer.on_video_recognition_task(frame_buffer)
	# if human_action is None:
	# 	return None
	prompt = base_prompt + '\n' + f'when the user is {human_action}'
	resp = await gpt_call(gpt_model_name, prompt)
	return post_process(resp)


async def run_main(model_name, prompt):
	resp = await gpt_call(model_name, prompt)
	poses = post_process(resp)
	print(poses)


# class ActionGenerator:
#
# 	def on_expression_generation_task(self, frame_buffer):
# 		human_action = video_recognizer.on_video_recognition_task(frame_buffer)
# 		if human_action is None:
# 			return None



if __name__ == "__main__":
	prompt = base_prompt + '\n' + 'when the user is walking the dog'
	# print(prompt)

	asyncio.run(run_main(gpt_model_name, prompt))
