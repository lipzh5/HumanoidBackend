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
from Utils import ImagePreprocess
from Utils.FrameBuffer import frame_buffer
from Utils.DialogueBuffer import diag_buffer
# from VisualModels.BLIP import blip_analyzer
from VisualModels.GPT4V import run_vqa_from_client_query
from VisualModels.Hiera import video_recognizer
from ActionGeneration import on_pose_generation
from VisualModels.InsightFace import find_from_db
from VisualModels.EmotionRec import emo_recognizer
from VisualModels.yolov7_main.fall_detect import fall_detector
from MultimodalModels.vle_model import vle_task_dealer
from Utils.OpenAIClient import client
# from LanguageModels.ChatModels import generate_ans, llama_to_cpu (penny note: use gpt-4o for answer polish instead)
from LanguageModels.RAG.main import RAGInfo  # for mq stuff training
import CONF
from Const import *
import time
import base64
from PIL import Image
import os.path as osp
import logging
log = logging.getLogger(__name__)
rag_info = RAGInfo(use_public_embedding=True, top_k=3)

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

ip = '10.6.34.99'   # dynamic ip of the robot
face_detect_addr = f'tcp://{ip}:6666'   # face detection result from Ameca
vsub_addr = f'tcp://{ip}:5000'  # From Ameca, 5000: mjpeg
# vsub_addr = 'tcp://10.126.110.67:5555'  # video capture data subscription
# vsub_sync_addr = 'tcp://10.126.110.67:5555'  # video capture data subscription
vtask_deal_addr = f'tcp://{ip}:2018' #'tcp://10.126.110.67:2006'
# vsub_mjpeg_addr = f'tcp://{ip}:5000'  # mjpeg From Ameca


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

async def on_video_reg_task(*args):  # TODO penny:  not in use
	return ResponseCode.Success, video_recognizer.on_video_recognition_task(frame_buffer)


async def on_pose_gen_task(*args):
	# llama_to_cpu()  # in case of cuda out of memory, delete this line if large memory is available
	ans = video_recognizer.on_video_rec_posegen_task(frame_buffer)
	# return ResponseCode.Success, ans (action, 'chat projects')
	response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": 
				"""You are a friendly humanoid robot named Ameca. Always provide brief but precise response to the user.
				Do mention the action provided by the user in your response, e.g., "I see you are reading a book, keep up the good job".
				"""},
                {"role": "user", "content": f"I am {ans[0]}"}
            ]
        )
	return ResponseCode.Success, (response.choices[0].message.content, ans[1])
	


async def on_face_rec_task(*args):
	try:
		force_recog = int(args[1]) if len(args) > 1 else True
		for i in range(CONF.face_reg_try_cnt):
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

async def on_fall_detection_task(*args):
	# response = '1'
	ans = fall_detector.detect(frame_buffer)
	response = '1' if ans else '0'
	return ResponseCode.Success, response


async def on_vle_task(*args):
	utterance, ts_end, duration, from_ameca = args
	print(f'ts_end :{ts_end}, duration: {duration} \n***** ')
	response_code, emo_anim = await vle_task_dealer.on_vle_task(utterance, ts_end, duration, from_ameca, diag_buffer)
	return response_code, emo_anim 



async def on_info_retrieve_task(*args):
	'''information retriece for staff training at MQ'''
	try:
		# print(args, type(args), args[0])
		ans = await rag_info.get_response(args[0].decode(CONF.encoding))
		return ResponseCode.Success, ans
	except Exception as e:
		print(f'****\n {str(e)} \n*****')
		return ResponseCode.Fail, None
	


TASK_DISPATCHER = {
	VisualTasks.VQA: on_vqa_task,
	VisualTasks.VideoRecognition: on_video_reg_task,
	VisualTasks.VideoRecogPoseGen: on_pose_gen_task,
	VisualTasks.FaceRecognition: on_face_rec_task,
	VisualTasks.EmotionImitation: on_emo_imitation_task,
	VisualTasks.FallDetection: on_fall_detection_task,
	NLPTask.RAG: on_info_retrieve_task, 
	MultimodalTasks.VLE: on_vle_task, 
}


class SubRouter:
	def __init__(self):
		super().__init__()
		self.sub_sock = ctx.socket(zmq.SUB)
		self.sub_sock.setsockopt(zmq.SUBSCRIBE, b'')
		self.sub_sock.setsockopt(zmq.CONFLATE, 1)

		self.face_detect_sub_sock = ctx.socket(zmq.SUB)
		self.face_detect_sub_sock.setsockopt(zmq.SUBSCRIBE, b'')
		# self.face_detect_sub_sock.setsockopt(zmq.CONFLATE, 1)  # do not use this flag, which will cause data loss

		try:
			self.sub_sock.connect(vsub_addr)
		except Exception as e:
			print('Check the ip of Ameca first!!!!')
			print('============================')
			print(str(e))
		try:
			self.face_detect_sub_sock.connect(face_detect_addr)
		except Exception as e:
			print(str(e))
			print('===='*3)
		# self.sub_sock.bind(vsub_addr)

		# context = zmq.Context.instance()
		# self.sub_sock_sync = context.socket(zmq.SUB)
		# self.sub_sock_sync.bind(vsub_sync_addr)
		# self.sub_sock_sync.setsockopt(zmq.SUBSCRIBE, b'')

		self.router_sock = ctx.socket(zmq.ROUTER)
		self.router_sock.connect(vtask_deal_addr)
		# self.router_sock.bind(vtask_deal_addr)

	# def sync_sub_vcap_data(self):
	# 	try:
	# 		while True:
	# 			msg = self.sub_sock_sync.recv_multipart()
	# 			frame_buffer.append_content(msg[0])
	# 			content_len = len(frame_buffer.buffer_content)
	# 			# if content_len % 64 == 0:
	# 			# 	print(f'sync len frame buffer content: {len(frame_buffer.buffer_content)}')
	# 	except Exception as e:
	# 		print(str(e))

	# def test_base64_encode(self, data):
	# 	encoded = base64.b64encode(data)
	# 	print('encode data successful ', len(data))
	# 	pass

	async def sub_face_detect_data(self):
		saved_faces = 0
		last_save_time = time.time()
		try:
			while True:
				data = await self.face_detect_sub_sock.recv_multipart()

				print('face detect data recvd: ', data)  # [xmin, ymin, width, height]
				print([float(t.decode(CONF.encoding)) for t in data])
				print(f'time.time: ', time.time())
				print('---'*5)
				if frame_buffer.buffer_content and saved_faces < 10 and 1 < time.time() - last_save_time: # TODO
					last_save_time = time.time()
					saved_faces += 1
					await self.debug_save_frame(frame_buffer.buffer_content[-1])

				'''
				face detect data recvd:  [b'0.025750000029802322', b'-0.024769999086856842', b'0.1606999933719635', b'0.2856999933719635', b'1711498877.46826']
				[0.025750000029802322, -0.024769999086856842, 0.1606999933719635, 0.2856999933719635, 1711498877.46826]
				time.time:  1711498877.444915
				'''
				# raw_data = await self.face_detect_sub_sock.recv()
				# print('raw data ', raw_data)
			pass
		except Exception as e:
			print(str(e))

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
						resp.extend([item.encode(CONF.encoding) for item in ans])
					else:
						resp.append(ans.encode(CONF.encoding))
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
			task_type = args[0].decode(CONF.encoding)
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
	# task = loop.create_task(sub_router.sub_face_detect_data())
	# await asyncio.gather(task)
	task1 = loop.create_task(sub_router.sub_vcap_data())
	task2 = loop.create_task(sub_router.route_visual_task())
	# task3 = loop.create_task(sub_router.sub_face_detect_data())
	await asyncio.gather(task1, task2)


if __name__ == "__main__":
	os.environ['TOKENIZERS_PARALLELISM']='false'
	asyncio.run(run_sub_router())




