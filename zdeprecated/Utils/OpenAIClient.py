# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: open ai client

from openai import AsyncOpenAI
import os.path as osp
file_path = osp.dirname(osp.abspath(__file__))
api_key = ''
# print(f'open ai cleint api key file path: {file_path} \n****')
try:
	with open(osp.join(file_path, "../openai_key.txt")) as fd:
		api_key = fd.read().strip()
		print('open ai key: ', api_key)
except FileNotFoundError:
	print('could not find open ai key file \n*****')

client = AsyncOpenAI(api_key=api_key)