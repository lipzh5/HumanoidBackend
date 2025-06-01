# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import zmq
import time

port = '5556'
# server is created with a socket type "zmq.REP" and is bound to well known port
context = zmq.Context()

socket = context.socket(zmq.REP)
socket.bind("tcp://10.126.110.67:%s" % port)


if __name__ == "__main__":
	# it will block on recv() to get a request before it can send a reply
	while True:
		message = socket.recv()
		print('received request: ', type(message), len(message))
		time.sleep(1)
		socket.send(b'server revd')
