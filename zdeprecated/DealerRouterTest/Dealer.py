# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

import asyncio
import sys

import zmq
import zmq.asyncio
from zmq.asyncio import Context
import os
from pathlib import Path
print('cwd: ', type(os.getcwd()))
print('dir: ', os.path.dirname(os.getcwd()))
import sys
# print(sys.path)
# print('dir name: ', Path(os.getcwd()).parent.name)
# Note: execute on terminal at directory "DealerRouterTest"
sys.path.append(os.path.dirname(os.getcwd()))

import os.path as osp
cur_dir = osp.abspath(osp.dirname(__file__)) # osp.dirname(os.getcwd())
sys.path.append(cur_dir)
print('cur dir: ', cur_dir)
sys.path.append(osp.join(cur_dir, '../'))

from Utils import ImagePreprocess
import time
from memory_profiler import profile

ctx = Context.instance()
url = 'tcp://127.0.0.1:3000'
pub_url = 'tcp://127.0.0.1:3001'
# pub and dealer (mimic request and data stream from Ameca)

img_bytes = ImagePreprocess.convert_img_to_stream()


class PubDealer:

	def __init__(self):
		self.deal_sock = ctx.socket(zmq.DEALER)
		self.deal_sock.setsockopt(zmq.IDENTITY, b'img_dealer')
		self.deal_sock.connect(url)

		self.pub_sock = ctx.socket(zmq.PUB)
		self.pub_sock.connect(pub_url)

	async def pub_img_data(self):
		try:
			while True:
				await self.pub_sock.send_multipart([img_bytes, str(time.time()).encode()])
				await asyncio.sleep(0.1)
		except Exception as e:
			print(str(e))
	# def pub_img_data(self):
	# 	try:
	# 		while True:
	# 			self.pub_sock.send_multipart([img_bytes, str(time.time()).encode()])
	# 			time.sleep(0.1)
	# 	except Exception as e:
	# 		print(str(e))

	async def deal_visual_task(self):
		try:
			while True:
				await self.deal_sock.send_multipart([b'VideoRecogPoseGen', b'1'])
				# await self.deal_sock.send_multipart([b'VQA', b'what is this in the picture?', str(time.time()).encode()])
				print(f'deal visual task sent111')
				msg = await self.deal_sock.recv_multipart()
				print(f'resp recvd222: ', msg)
				await asyncio.sleep(1.)
				# while True:
				# 	await asyncio.sleep(1.)
				#
		except Exception as e:
			print('error with img data dealer')
			print(str(e))



async def img_data_dealer():
	deal_sock = ctx.socket(zmq.DEALER)
	deal_sock.setsockopt(zmq.IDENTITY, b'img_dealer')
	deal_sock.connect(url)
	print('Img data dealer initialized!!')
	try:
		while True:
			await asyncio.sleep(1.)
			await deal_sock.send_multipart([img_bytes])
			msg = await deal_sock.recv_multipart()
			print(f'resp recvd: ', msg)

	except Exception as e:
		print('error with img data dealer')
		print(str(e))


@profile
async def run_main():
	pub_dealer = PubDealer()
	loop = asyncio.get_event_loop()
	task1 = loop.create_task(pub_dealer.pub_img_data())
	task2 = loop.create_task(pub_dealer.deal_visual_task())
	await asyncio.gather(task1, task2)


if __name__ == "__main__":
	pub_dealer = PubDealer()
	asyncio.run(pub_dealer.deal_visual_task())
	# asyncio.run(run_main())
	# loop = asyncio.get_event_loop()
	# corou = loop.create_task(pub_dealer.deal_visual_task())

	# asyncio.run(img_data_dealer())

