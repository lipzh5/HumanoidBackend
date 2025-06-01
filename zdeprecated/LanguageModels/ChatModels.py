# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

from llama import Dialog, Llama

CKPT_DIR = '/home/penny/pycharmprojects/AmecaBackend/LanguageModels/llama3/Meta-Llama-3-8B-Instruct/'
TOKENIZER_PATH = CKPT_DIR + 'tokenizer.model'
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 6

generator = Llama.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
    )

def generate_ans(task_type, raw_answer, query=''):
	generator.model.cuda()
	if raw_answer is None:
		return None
	print(f'chat model raw msg: {raw_answer} \n ***************')
	if task_type == 'face_recognition':
		dialog = [
				{"role": "system", "content": "You are a friendly humanoid robot named Ameca. Reply briefly with no more than 3 sentences"},
				{"role": "user", "content": f"I am {raw_answer}."}] 
	elif task_type == 'vqa':
		dialog =  [
			{"role": "system", "content": f"""
			You are an intelligent AI assistant, always ready to polish the answer to the visual-question: {query} and reply to the user with the polished answer briefly.
			Do mention the word provided by the user in your response.
			"""
			},
			{"role": "user", f"content": f"{raw_answer}."}
		]
	elif task_type == 'action_recognition':
		dialog = [
			{"role": "system", "content": """
			You are a friendly humanoid robot named Ameca. Always provide brief but precise response to the user.
			Do mention the action provided by the user in your response.
			"""
			},
			{"role": "user", "content": f"I am {raw_answer}."}
		]

	dialogs: List[Dialog] = [dialog,]
	results = generator.chat_completion(
		dialogs,
		max_gen_len=None,
		temperature=0.6,
		top_p=0.9,
	)
	return results[0]['generation']['content']

def llama_to_cpu():
	generator.model.to('cpu')



def main(
	ckpt_dir: str = '/home/penny/pycharmprojects/AmecaBackend/LanguageModels/llama3/Meta-Llama-3-8B-Instruct/',
	tokenizer_path: str = '/home/penny/pycharmprojects/AmecaBackend/LanguageModels/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model',
	temperature: float = 0.6,
	top_p: float = 0.9,
	max_seq_len: int = 512,
	max_batch_size: int = 4,
	max_gen_len: Optional[int] = None,
):
	"""
	Examples to run with the models finetuned for chat. Prompts correspond of chat
	turns between the user and assistant with the final one always being the user.

	An optional system prompt at the beginning to control how the model should respond
	is also supported.

	The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

	`max_gen_len` is optional because finetuned models are able to stop generations naturally.
	"""
	# generator = Llama.build(
	# 	ckpt_dir=ckpt_dir,
	# 	tokenizer_path=tokenizer_path,
	# 	max_seq_len=max_seq_len,
	# 	max_batch_size=max_batch_size,
	# )

	dialogs: List[Dialog] = [
		# [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
		# [
#             {"role": "user", "content": "I am going to Paris, what should I see?"},
#             {
#                 "role": "assistant",
#                 "content": """\
# Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

# 1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
# 2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
# 3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

# These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
#             },
#             {"role": "user", "content": "What is so great about #1?"},
#         ],
		[
			{"role": "system", "content": "You are a friendly humanoid robot named Ameca. Reply briefly with no more than 3 sentences"},
			{"role": "user", "content": "I am Penny."}
			# {"role": "system", "content": "Always answer with Haiku"},
			# {"role": "user", "content": "I am going to Paris, what should I see?"},
		],
		[
			{"role": "system", "content": """
			You are an intelligent AI assistant, always ready to polish the answer to the visual-question 'what is this in my hand?' and reply to the user with the polished answer briefly.
			Do mention the word provided by the user in your response.
			"""
			},
			{"role": "user", "content": "bottle."}, 

			# {"role": "system", "content": "Always answer with Haiku"},
			# {"role": "user", "content": "I am going to Paris, what should I see?"},
		],
		[
			{"role": "system", "content": """
			You are a friendly humanoid robot named Ameca. Always provide brief but precise response to the user.
			Do mention the action provided by the user in your response.
			"""
			
			},
			{"role": "user", "content": "I am playing badminton."}
			# {"role": "system", "content": "Always answer with Haiku"},
			# {"role": "user", "content": "I am going to Paris, what should I see?"},
		],
		# [
		#     {
		#         "role": "system",
		#         "content": "Always answer with emojis",
		#     },
		#     {"role": "user", "content": "How to go from Beijing to NY?"},
		# ],
	]
	results = generator.chat_completion(
		dialogs,
		max_gen_len=max_gen_len,
		temperature=temperature,
		top_p=top_p,
	)

	for dialog, result in zip(dialogs, results):
		for msg in dialog:
			print(f"{msg['role'].capitalize()}: {msg['content']}\n")
		print(
			f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
		)
		print("\n==================================\n")

def test_vllm():
	from openai import OpenAI
	client = OpenAI(
		base_url="http://localhost:8000/v1",
		api_key="token-abc123",
	)

	completion = client.chat.completions.create(
	model="NousResearch/Meta-Llama-3-8B-Instruct",
	messages=[
		{"role": "user", "content": "Hello!"}
	]
	)

	print(completion.choices[0].message)


if __name__ == "__main__":
	test_vllm()
	# fire.Fire(main)
