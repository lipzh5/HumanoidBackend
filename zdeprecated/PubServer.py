# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import zmq
import random
import sys
import time

url = 'tcp://10.126.110.67:5556'
ctx = zmq.Context()
socket = ctx.socket(zmq.PUB)
socket.bind(url)


def send():
	while True:
		topic = random.randrange(9999, 10005)
		messagedata = random.randrange(1, 215) - 80
		print('message data: ', messagedata)
		socket.send(f'{topic},{messagedata}'.encode('utf-8'))
		time.sleep(1)


if __name__ == "__main__":
	send()
