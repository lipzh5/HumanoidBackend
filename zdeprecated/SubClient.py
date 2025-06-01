# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import sys
import zmq

# url = 'tcp://10.126.110.67:5556'
url = 'tcp://10.6.33.44:5556'  # host ip

ctx = zmq.Context()
socket = ctx.socket(zmq.SUB)
socket.connect(url)


class FrameBuffer:
	content = b''
	n_frames = 0

	@classmethod
	def append_content(cls, val):
		cls.content += val
		cls.n_frames += 1


def subscribing():
	topic_filter = b''   #'10001'.encode('utf-8')
	socket.setsockopt(zmq.SUBSCRIBE, topic_filter)

	# process 5 updates
	while True:
		msg = socket.recv()
		FrameBuffer.append_content(msg)


if __name__ == "__main__":
	subscribing()
