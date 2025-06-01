# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: subscribe video capture data from Ameca

import asyncio
import io

import zmq
import zmq.asyncio
from const import *
import CONF
from VisualModels.BLIP import blip_analyzer
from VisualModels.Hiera import video_recognizer
from Utils.FrameBuffer import FrameBuffer
import ActionGeneration as ag

# NOTE: check url if no message recvd
subscribe_url = 'tcp://10.6.33.4:5556'   # Ameca host ip
send_response_url = 'tcp://*:2000'  # local ip # 10.126.110.67


context = zmq.asyncio.Context()

visual_tasks = set()
resp_sending_tasks = set()
frame_buffer = FrameBuffer()  # TODO add a wrapper may be better to replace this global var


def on_vqa_task(*args):
	if CONF.debug:
		ts = args[-1]
		print(f'on vqa task timestamp: {ts}')
	frame_data = frame_buffer.consume_one_frame()
	if frame_data:
		return blip_analyzer.on_vqa_task(frame_data, args[0])  # *args[:2]
	return None


def on_video_rec_task(*args):
	if CONF.debug:
		ts = args[-1]
		print(f'on video rec task timestamp: {ts}')
	return video_recognizer.on_video_recognition_task(frame_buffer)


def on_video_rec_pose_gen_task(*args):
	action = on_video_rec_task(*args)
	if action is None:
		return None
	prompt = CONF.base_prompt + '\n' + f'when the user is {action}'
	gpt_completion = ag.gpt_call(CONF.gpt_model_name, prompt)
	text = gpt_completion.choices[0].text
	text = text.replace('[', '*').replace(']', '*').split('*')
	print('gpt generated text: ', text[1])
	poses = text[1].split(',')
	poses = [pose.lstrip().rstrip() for pose in poses]
	return poses


def on_face_rec_task(*args):
	pass


def on_vcap_data_recvd(*args):
	frame_buffer.append_content(args[0])
	pass


TASK_DISPATCHER = {
	VisualTasks.PureData: on_vcap_data_recvd,
	VisualTasks.VQA: on_vqa_task,
	VisualTasks.VideoRecognition: on_video_rec_task,
	VisualTasks.FaceRecognition: on_face_rec_task,
	VisualTasks.VideoRecogPoseGen: on_video_rec_pose_gen_task,
}


def run_background_visual_task(msg, full_cb):

	async def run_task(msg, callback):  # msg: [task_type, img_bytes, xx ]
		print('run task!!! ')
		encoding = CONF.encoding
		task_type = msg[0].decode(encoding)
		print('task type: ', task_type)
		response = TASK_DISPATCHER[task_type](*msg[1:])
		print(f'on task finish response : {type(response), response}')
		if response is None:
			return
		params = [resp.encode(encoding) for resp in response] if isinstance(response, list) else [response.encode(encoding)]
		# if CONF.debug:
		# 	params.append(msg[-1])
		# params = (response.encode(encoding), msg[-1]) if CONF.debug else (response, )
		callback(*params)
		# await asyncio.sleep(1.)

	loop = asyncio.get_event_loop()

	coroutine = loop.create_task(
		run_task(msg, full_cb)
	)
	visual_tasks.add(coroutine)
	coroutine.add_done_callback(lambda _:visual_tasks.remove(coroutine))
	print(f'len of visual tasks: {len(visual_tasks)}')


# ==============================
# TODO should organize these sockets in a tidy way
resp_socket = context.socket(zmq.PUB)
resp_socket.bind(send_response_url)


def on_task_finish_cb(*args, timestamp=b''):  # response has already been encoded
	"""should send response to Robot, when visual task is finished """
	async def send_resp_to_robot(*args, ts=b''):
		await resp_socket.send_multipart([*args, ts])

	loop = asyncio.get_event_loop()
	coroutine = loop.create_task(send_resp_to_robot(*args, ts=timestamp))
	resp_sending_tasks.add(coroutine)
	coroutine.add_done_callback(lambda _: resp_sending_tasks.remove(coroutine))
	# print(f'msg has been processed!!! {response}')


async def run_sub():
	socket = context.socket(zmq.SUB)
	# we can connect to several endpoints if we desire, and receive from all
	socket.connect(subscribe_url)
	socket.setsockopt(zmq.SUBSCRIBE, b'')
	print('video cap subscriber initialized!!!!')
	while True:
		msg = await socket.recv_multipart()
		# print('len msg: ', len(msg))
		run_background_visual_task(msg, on_task_finish_cb)


ctx = zmq.Context()

def run_sub_sync():
	socket = ctx.socket(zmq.SUB)
	socket.connect(subscribe_url)
	socket.setsockopt(zmq.SUBSCRIBE, b'')
	print('Video cap subscriber sync initialized!!!')
	while True:
		msg = socket.recv_multipart()
		task_type = msg[0].decode(CONF.encoding)
		print('task type: ', task_type)
		if task_type == VisualTasks.PureData:
			on_vcap_data_recvd(msg[1])
		else:
			print(f'visual task: {task_type}')
			run_background_visual_task(msg, on_task_finish_cb)
		# print(f'msg recvd: {len(msg[0])}')


# def func(*args):
# 	print(args)


if __name__ == "__main__":
	# msg = ('type', )
	# t = msg[1:]
	# print(t)
	# func(*t)
	run_sub_sync()
	# asyncio.run(run_sub())

