# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import asyncio
import time
import zmq
from zmq.asyncio import Context, Poller

url = 'tcp://10.126.110.67:5555'
ctx = Context.instance()


async def ping() -> None:
	"""print dots to indicate idleness"""
	while True:
		await asyncio.sleep(0.5)
		print('.')


async def receiver() -> None:
	"""receives messages with polling"""
	pull = ctx.socket(zmq.PULL)
	pull.connect(url)
	poller = Poller()
	poller.register(pull, zmq.POLLIN)
	while True:
		events = await poller.poll()
		if pull in dict(events):
			print('receiving ', events)
			msg = await pull.recv_multipart()
			print('recvd ', type(msg), len(msg), len(msg[0]))


async def sender() -> None:
	"""send a message every second"""
	tic = time.time()
	push = ctx.socket(zmq.PUSH)
	push.bind(url)
	while True:
		print('sending')
		await push.send_multipart([str(time.time()-tic).encode('utf-8'), 'hellow'.encode('utf-8')])
		await asyncio.sleep(1)


if __name__ == "__main__":
	asyncio.run(receiver())
	# asyncio.run(asyncio.wait([
	# 	ping(),
	# 	receiver(),
	# 	sender()
	# ]))












