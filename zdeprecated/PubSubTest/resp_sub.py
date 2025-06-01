# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

import asyncio
import zmq
import zmq.asyncio

context = zmq.asyncio.Context()


async def run_resp_sub():
	socket = context.socket(zmq.SUB)
	socket.connect('tcp://127.0.0.1:20001')
	socket.setsockopt(zmq.SUBSCRIBE, b'')
	print('response sub initialized!!!')
	while True:
		msg = await socket.recv_multipart()
		print('robot response: ', msg)
		print(f'robot response: {msg[0].decode("utf-8")}, timestamp: {msg[-1]}')

if __name__ == "__main__":
	asyncio.run(run_resp_sub())