# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import asyncio
import zmq
from zmq.asyncio import Context
import time

resp_back_url = 'tcp://127.0.0.1:5556'
ctx = Context.instance()


class RespReceiver:
	def __init__(self):
		self.resp_back_sock = ctx.socket(zmq.PULL)
		self.resp_back_sock.connect(resp_back_url)

	async def recv_resp(self):
		while True:
			msg = await self.resp_back_sock.recv_multipart()
			print('recv resp!!!!', msg)


resp_recvr = RespReceiver()

if __name__ == "__main__":
	asyncio.run(resp_recvr.recv_resp())

