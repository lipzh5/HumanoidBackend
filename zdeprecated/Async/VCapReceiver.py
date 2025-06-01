# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import asyncio
import time
import zmq
from zmq.asyncio import Context

ZMQ_URL = 'tcp://10.6.33.70:5557'  # host ip


async def receiver():
	ctx = Context.instance()
	sub_socket = ctx.socket(zmq.SUB)
	port = '5557'
	# sub_socket.bind("tcp://*:%s" % port) # ZMQ_URL
	sub_socket.connect(ZMQ_URL)  # should be "connect" not "bind"
	sub_socket.setsockopt(zmq.SUBSCRIBE, b'')
	print('VCap Data Subscriber Initialized Successfully!')
	try:
		while True:
			msg = await sub_socket.recv_multipart()
			print(type(msg[0]))
			print(f'len of msg :{len(msg[0])}')
	except Exception as e:
		print(str(e))


if __name__ == "__main__":
	asyncio.run(receiver())
