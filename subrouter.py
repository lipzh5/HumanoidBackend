# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import sys
print(f'sys.path: {sys.path}')
import asyncio
# import io
from io import BytesIO
from collections import defaultdict
import os

import zmq
import zmq.asyncio
from zmq.asyncio import Context
from utils.framebuffer import frame_buffer
from utils.openaiclient import client
from custommodels.gpt4v import run_vqa_from_client_query
from custommodels.hiera import video_recognizer
from custommodels.insightface import find_from_db
from custommodels.emotionrec import emo_recognizer

# from LanguageModels.ChatModels import generate_ans, llama_to_cpu (penny note: use gpt-4o for answer polish instead)
# from LanguageModels.RAG.main import RAGInfo  # for mq stuff training
from conf import *
from const import *
import time
import base64
from PIL import Image
import os.path as osp
import logging
log = logging.getLogger(__name__)
# rag_info = RAGInfo(use_public_embedding=True, top_k=3)

# conda activate amecabackend
# torchrun --nproc_per_node 1 AmecaSubRouter.py

# note: video capture needs configure video_capture node as follows
'''
{
  "data_address": "tcp://0.0.0.0:5001",
  "mjpeg_address": "tcp://0.0.0.0:5000",
  "sensor_name": "Left Eye Camera",
  "video_device": "/dev/eyeball_camera_left",
  "video_height": 720,
  "video_width": 1280
}
'''

# ip = '10.6.36.39'   # dynamic ip of the robot
# # face_detect_addr = f'tcp://{ip}:6666'   # face detection result from Ameca
# vsub_addr = f'tcp://{ip}:5000'  # From Ameca, 5000: mjpeg
# vtask_deal_addr = f'tcp://{ip}:2017' #'tcp://10.126.110.67:2006'
# # vsub_mjpeg_addr = f'tcp://{ip}:5000'  # mjpeg From Ameca


ctx = Context.instance()

LAST_QUERY_TS = defaultdict(float)
MIN_QUERY_INTERVAL = 2.0


def is_valid_query(task_type):
	"""1. avoid queries with extremely high frequency"""
	# last_ts = LAST_QUERY_TS.get(VisualTasks.VQA, 0)
	now = time.time()
	valid = now - LAST_QUERY_TS.get(task_type, 0) > MIN_QUERY_INTERVAL
	LAST_QUERY_TS[task_type] = now 
	return valid

async def on_vqa_task(*args):
	frame = frame_buffer.consume_one_frame()  # TODO merge to blip_analyzer
	if not frame:
		return ResponseCode.Fail, None
	res = await run_vqa_from_client_query(frame, *args)
	return ResponseCode.Success, res
	# print(f'args: {args} \n *******')
	# ans = blip_analyzer.on_vqa_task(frame, *args)
	# return ResponseCode.Success, generate_ans('vqa', ans, query=args[0].decode(encoding=CONF.encoding))

async def on_video_reg_task(*args): 
	return ResponseCode.Success, video_recognizer.on_video_recognition_task(frame_buffer)



async def on_face_rec_task(*args):
	try:
		force_recog = int(args[1]) if len(args) > 1 else True
		for i in range(FACE_REC_TRY_CNT):
			res_code, found = find_from_db(frame_buffer.buffer_content[-i-1], ignore_ts=force_recog)
			if res_code != ResponseCode.Success:
				continue
			response = await client.chat.completions.create(
				model="gpt-4o",
				messages=[
					{"role": "system", "content": "You are a friendly humanoid robot named Ameca. Reply briefly with no more than 3 sentences"},
					{"role": "user", "content": f"I am {found}"}
				]
			)
			return res_code, response.choices[0].message.content 
		return ResponseCode.Fail, None
			
	except Exception as e:
		print(str(e))
		print(f'-------------')
		import traceback
		traceback.print_stack()
		return ResponseCode.Fail, None

async def on_emo_imitation_task(*args):
	print(f'emotion recognition task!!! {args} \n ****')
	frame = frame_buffer.consume_one_frame()
	if not frame:
		return ResponseCode.Fail, None
	try:
		response = await emo_recognizer.on_emotion_recog_task(frame)
		return ResponseCode.Success, response  # (emo_anim, anslysis)
	except Exception as e:
		print(str(e))
		print('==============')
		import traceback
		traceback.print_stack()
		return ResponseCode.Fail, None




TASK_DISPATCHER = {
	CustomTasks.VQA: on_vqa_task,
	CustomTasks.VideoRecognition: on_video_reg_task,
	CustomTasks.FaceRecognition: on_face_rec_task,
	CustomTasks.EmotionImitation: on_emo_imitation_task,
}


class SubRouter:
	def __init__(self):
		super().__init__()
		self.sub_sock = ctx.socket(zmq.SUB)
		self.sub_sock.setsockopt(zmq.SUBSCRIBE, b'')
		self.sub_sock.setsockopt(zmq.CONFLATE, 1)

		# self.face_detect_sub_sock = ctx.socket(zmq.SUB)
		# self.face_detect_sub_sock.setsockopt(zmq.SUBSCRIBE, b'')
		# self.face_detect_sub_sock.setsockopt(zmq.CONFLATE, 1)  # do not use this flag, which will cause data loss

		try:
			self.sub_sock.connect()
		except Exception as e:
			print('Check the ip of Ameca first!!!!')
			print('============================')
			print(str(e))
		# self.sub_sock.bind(vsub_addr)

		# context = zmq.Context.instance()
		# self.sub_sock_sync = context.socket(zmq.SUB)
		# self.sub_sock_sync.bind(vsub_sync_addr)
		# self.sub_sock_sync.setsockopt(zmq.SUBSCRIBE, b'')

		self.router_sock = ctx.socket(zmq.ROUTER)
		self.router_sock.connect(TASK_DEALER_ADDR)
		# self.router_sock.bind(vtask_deal_addr)


	async def debug_save_frame(self, frame_data: bytes):
		img = Image.open(BytesIO(frame_data))
		# img = Image.frombytes('RGB', IMG_SIZE, frame_data)
		img_dir = 'Assets/debug_imgs'
		if not osp.exists(img_dir):
			os.makedirs(img_dir)
		img.save(osp.join(img_dir, f'{time.time()}.png' ))

	async def sub_vcap_data(self):
		# start_time = 1733097054.5939214 + 60 * 5
		# end_time = 1733097054.5939214 + 60 * 5 + 10
		try:
			ts = time.time()
			cnt = 0
			while True:
				data = await self.sub_sock.recv()
				frame_buffer.append_content(data)  # TODO time.time() , img = Image.open(BytesIO(data))
				# print(f'subscrive vcap data!!! {len(frame_buffer.buffer_content)}')
				# NOTE: debug save
				# now = time.time()
				# if ts + 6 < now < ts + 75:
				# 	await self.debug_save_frame(frame_buffer.buffer_content[-1])
				# time_stamp = time.time()
				
				# if cnt < 1:
				# 	img = Image.frombytes('RGB', (1280, 720), data)
				# 	img.save('jpeg_img_from_bytes.png')
				# cnt += 1
				# if cnt == 60:
				# 	total = time.time() - ts
				# 	print(f'total time for receiving 60 frames is {total}')

		except Exception as e:
			print(str(e))

	async def route_visual_task(self):
		try:
			while True:
				msg = await self.router_sock.recv_multipart()
				identity = msg[0]
				print('route visual task identity: ', identity)
				try:
					res_code, ans = await self.deal_visual_task(*msg[1:])
					if ans is None:
						ans = 'None'
					print(f'task answer:{ans} \n ------- ')
					resp = [identity, res_code]
					if isinstance(ans, list) or isinstance(ans, tuple):
						resp.extend([item.encode(ENCODING) for item in ans])
					else:
						resp.append(ans.encode(ENCODING))
				except Exception as e:
					print(str(e))
					print(f'msg: {msg}')
					print('----------')
					resp = [identity, ResponseCode.Fail, b'None']

				await self.router_sock.send_multipart(resp)
				
		except Exception as e:
			print(str(e))
			print('-----router visual task line 293----')
			import traceback
			traceback.print_stack()

	async def deal_visual_task(self, *args):
		try:
			# ts = time.time()
			task_type = args[0].decode(ENCODING)
			# if not is_valid_query(task_type):
			# 	print(f'invalid query!!!! {task_type} \n **************')
			# 	return (ResponseCode.KeepSilent, None)
			ans = await TASK_DISPATCHER[task_type](*args[1:])
			# print(f'inference time for {task_type}: {time.time()-ts}') # around 0.05s
			# ans = blip_analyzer.on_vqa_task(frame, args[1], debug=CONF.debug)
			# print('deal visual task ans: ', ans)
			return ans
		except Exception as e:
			print(str(e))
			return ResponseCode.Fail, None
		# return TASK_DISPATCHER[task_type](frame, *tuple(map(lambda x: x.decode(AmecaCONF.encoding), args[1:])))
		# return blip_analyzer.on_vqa_task(frame, args[1].decode(), debug=AmecaCONF.debug)


async def run_sub_router():
	sub_router = SubRouter()
	loop = asyncio.get_event_loop()
	# await asyncio.gather(task)
	task1 = loop.create_task(sub_router.sub_vcap_data())
	task2 = loop.create_task(sub_router.route_visual_task())
	await asyncio.gather(task1, task2)


if __name__ == "__main__":
	os.environ['TOKENIZERS_PARALLELISM']='false'
	asyncio.run(run_sub_router())




