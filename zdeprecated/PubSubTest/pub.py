# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: publisher subscriber test (mimic real-time data processing)

import asyncio
import datetime
import zmq
import zmq.asyncio

from Utils import ImagePreprocess
from Const import *
import CONF
import random
import time

context = zmq.asyncio.Context()

# b64 encode image on debug mode
img_bytes = ImagePreprocess.convert_img_to_stream()


async def run_server(host, port):
	socket = context.socket(zmq.PUB)
	socket.bind(f'tcp://{host}:{port}')  # use "bind" for publisher and "connect" for subscriber
	print('publisher initialized!!!')
	encoding = CONF.encoding
	while True:
		await asyncio.sleep(1.)
		query = 'what is this in my hand'.encode(encoding)
		# r = random.randint(0, 10)
		task_type = VisualTasks.VideoRecogPoseGen.encode(encoding)
		data = [task_type, img_bytes]
		# if r % 2 == 0:
		# 	task_type = VisualTasks.VQA.encode(encoding)
		# 	data = [task_type, img_bytes, query]
		# else:
		# 	task_type = VisualTasks.VideoRecognition.encode(encoding)
		# 	data = [task_type, img_bytes]
		print(f'task type: {task_type}, send img len in bytes: {len(img_bytes)}')
		# task_type = VisualTasks.VQA.encode(encoding)
		# data = [task_type, img_bytes, query]
		if CONF.debug:
			data.append(str(datetime.datetime.now()).encode(encoding))
		await socket.send_multipart(data)

		# await socket.send_multipart([str(datetime.datetime.now()).encode()])


# updated pure vcap data publish
def run_server_sync(host, port):
	ctx = zmq.Context()
	socket = ctx.socket(zmq.PUB)
	socket.bind(f'tcp://{host}:{port}')  # use "bind" for publisher and "connect" for subscriber
	print('publisher initialized!!!')
	encoding = CONF.encoding
	while True:
		time.sleep(0.1)
		data = [img_bytes, str(datetime.datetime.now()).encode(encoding)]
		socket.send_multipart(data)


if __name__ == "__main__":
	run_server_sync('127.0.0.1', 2000)
	# import os
	# print(os.getcwd())
	# asyncio.run(run_server('127.0.0.1', 2000))


