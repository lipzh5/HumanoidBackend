# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
from collections import deque
import CONF


class FrameBuffer:
	_instance = None
	buffer_content = deque()

	def __new__(cls, *args, **kwargs):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance

	@classmethod
	def append_content(cls, frame: bytes):
		while len(cls.buffer_content) >= CONF.frame_buffer_max_len:
			cls.buffer_content.popleft()
		cls.buffer_content.append(frame)

	@classmethod
	def consume_content(cls, num):
		if num > len(cls.buffer_content):
			return None
		return [cls.buffer_content.popleft() for _ in range(num)]

	@classmethod
	def consume_one_frame(cls):
		if cls.buffer_content:
			return cls.buffer_content[-1]
			# return cls.buffer_content.popleft()
		return None


frame_buffer = FrameBuffer()



if __name__ == "__main__":
	d = deque()
	if d:
		print(d.popleft())
	else:
		print('d is none')

	# buffer = FrameBuffer()
	# buffer2 = FrameBuffer()
	# print(f'buffer content: {buffer.buffer_content}')
	# print(f'class :{FrameBuffer.buffer_content}')
	# buffer.append_content(b'hello world!!!')
	# buffer2.append_content(b'how are you')
	#
	# print(buffer2.buffer_content)
	# print(FrameBuffer.buffer_content)
