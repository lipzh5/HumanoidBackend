# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import asyncio
import zmq
from zmq.asyncio import Context
import time
from Utils import ImagePreprocess

# b64 encode image on debug mode
img_bytes = ImagePreprocess.convert_img_to_stream()

url = 'tcp://127.0.0.1:5555'
ctx = Context.instance()


class Sender:
	def __init__(self):
		self.enabled = True
		self.push_sock = ctx.socket(zmq.PUSH)
		self.push_sock.bind(url)
		# self.listening_task = None
		# self.start_listen_to_resp()


	# def start_listen_to_resp(self):
	# 	async def listen_to_resp():
	# 		while True:
	# 			msg = self.resp_back_sock.recv_multipart()
	# 			print('resp back msg: ', msg)
	# 			print('=======')
	#
	# 	loop = asyncio.get_event_loop()
	# 	self.listening_task = loop.create_task(listen_to_resp())

	async def send_data(self):
		await self.push_sock.send_multipart([str(time.time()).encode()])
		await asyncio.sleep(1.)
		# while True:
		# 	if not self.enabled:
		# 		print('sending not enabled!!! ')
		# 		continue
		# 	print('sending success222')
		# 	await self.push_sock.send_multipart([str(time.time()).encode()])
		# 	await asyncio.sleep(1.)


sender = Sender()

async def send_data():
	push_sock = ctx.socket(zmq.PUSH)
	push_sock.bind(url)
	for i in range(10):
		print('sending success', i)
		await push_sock.send_multipart([str(time.time()).encode()])
		# await asyncio.sleep(1.0)
		pass
	# while True:
	# 	print('sending success222')
	# 	await push_sock.send_multipart([str(time.time()).encode()])
	# 	await asyncio.sleep(1.0)

def send_data_wrapper():
	while True:
		time.sleep(0.5)
		data = img_bytes
		loop = asyncio.get_event_loop()
		loop.create_task(sender.send_data())


async def test1():
	await asyncio.sleep(0.5)
	print('test1 awake ')


async def test2():
	await asyncio.sleep(0.5)
	print('test2 awake')


async def run():
	loop = asyncio.get_event_loop()
	tasks = [loop.create_task(test1()), loop.create_task(test2())]
	await asyncio.gather(*tasks)


if __name__ == "__main__":
	asyncio.run(run())
	# loop = asyncio.get_event_loop()
	# tasks = [loop.create_task(test1()), loop.create_task(test2())]
	# await asyncio.gather(*tasks)
	# loop.create_task([test1(), test2()])
	# asyncio.run(sender.send_data())




