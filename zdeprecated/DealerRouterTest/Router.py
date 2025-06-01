# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import asyncio
import zmq
import zmq.asyncio
from zmq.asyncio import Context
# Note: execute on terminal at directory "DealerRouterTest"
import sys, os
import os.path as osp
cur_dir = osp.abspath(osp.dirname(__file__)) # osp.dirname(os.getcwd())
sys.path.append(cur_dir)
print('cur dir: ', cur_dir)
sys.path.append(osp.join(cur_dir, '../'))


from Utils import ImagePreprocess
from Utils.FrameBuffer import frame_buffer
from VisualModels.BLIP import blip_analyzer
from VisualModels.Hiera import video_recognizer
from memory_profiler import profile

ctx = Context.instance()
url = 'tcp://127.0.0.1:3000'
sub_url = 'tcp://127.0.0.1:3001'
# sub and router


class SubRouter:

	def __init__(self):
		self.router_sock = ctx.socket(zmq.ROUTER)
		self.router_sock.bind(url)

		self.sub_sock = ctx.socket(zmq.SUB)
		self.sub_sock.bind(sub_url)
		self.sub_sock.setsockopt(zmq.SUBSCRIBE, b'')

	@profile
	async def sub_img_data(self):
		try:
			while True:
				msg = await self.sub_sock.recv_multipart()
				frame_buffer.append_content(msg[0])
				# print(f'sub recvd at {msg[1].decode()}')
		except Exception as e:
			print(str(e))

	# def sub_img_data(self):
	# 	try:
	# 		while True:
	# 			msg = self.sub_sock.recv_multipart()
	# 			frame_buffer.append_content(msg[0])
	# 			print(f'sub recvd at {msg[1].decode()}')
	# 	except Exception as e:
	# 		print(str(e))


	async def deal_query(self, *args): # TODO replace with actual deal method
		# print(f'backend deal: arg0: {args[0]}, args1: {args[1]}')
		# await asyncio.sleep(1.)
		# TODO dispatch task based on different task type
		# query = 'please describe this picture'
		return video_recognizer.on_video_rec_posegen_task(frame_buffer)
		# return video_recognizer.on_video_recognition_task(frame_buffer)
		# frame = frame_buffer.consume_one_frame()
		# if not frame:
		# 	return None
		# print(f'type of args[0] {type(args[0])}')
		# return blip_analyzer.on_vqa_task(frame, args[0], debug=True) # f'answer {args[1].decode()}'

	@profile
	async def route_visual_task(self):
		try:
			while True:
				msg = await self.router_sock.recv_multipart()
				identity = msg[0]
				print(f'\n****\n msg: {msg}\n***')
				print(f'msg -1: {msg[-1]}, {type(msg[-1])}, int msg-1: {int(msg[-1])}, int? {int(msg[-1])==1}')
				print(f'msg revd: {len(msg)}', msg)
				ans = await self.deal_query(*msg[1:])
				if ans is None:
					ans = 'None'
					print('ans is None')
				print('route visual task ans: ', ans)
				print('====')
				await self.router_sock.send_multipart([identity, ans.encode()])

		except Exception as e:
			print('something wrong with img data router')
			print(str(e))



async def img_data_router():
	router_sock = ctx.socket(zmq.ROUTER)
	router_sock.bind(url)
	print('img data router initialized!!!')
	try:
		while True:
			msg = await router_sock.recv_multipart()
			print(f'msg revd: {len(msg)}, msg o: {msg[0]}')
			identity = msg[0]
			await router_sock.send_multipart([msg[0], b'ans'])
	except Exception as e:
		print('something wrong with img data router')
		print(str(e))


@profile
async def run_main():
	sub_router = SubRouter()
	# sub_router.sub_img_data()
	loop = asyncio.get_event_loop()
	task1 = loop.create_task(sub_router.sub_img_data())
	task2 = loop.create_task(sub_router.route_visual_task())
	await asyncio.gather(task1, task2)



if __name__ == "__main__":
	sub_router = SubRouter()
	asyncio.run(sub_router.route_visual_task())
	# asyncio.run(img_data_router())
	# asyncio.run(run_main())