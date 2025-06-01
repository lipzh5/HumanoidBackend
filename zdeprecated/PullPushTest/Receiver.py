# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import asyncio
import zmq
from zmq.asyncio import Context

url = 'tcp://127.0.0.1:5555'
ctx = Context.instance()

push_url = 'tcp://127.0.0.1:5556'


class Receiver:
	def __init__(self):
		self.enabled = True
		self.pull_sock = ctx.socket(zmq.PULL)
		self.pull_sock.connect(url)
		self.push_sock = ctx.socket(zmq.PUSH)
		self.push_sock.bind(push_url)

	async def recv_data(self):
		while True:
			if not self.enabled:
				print('recv not enabled!!!!')
			msg = await self.pull_sock.recv_multipart()
			print('recvd msg: ', msg)
			msg.append(b'hhh')
			await self.push_sock.send_multipart(msg)


recvr = Receiver()


def receiver():
	context = zmq.Context()
	pull_sock = context.socket(zmq.PULL)
	pull_sock.connect(url)
	print('receiver initialized successfully!!')
	while True:
		msg = pull_sock.recv_multipart()
		print('recvd: ', msg)


if __name__ == "__main__":
	asyncio.run(recvr.recv_data())