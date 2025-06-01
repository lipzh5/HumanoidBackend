# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import asyncio
import io

import zmq
import zmq.asyncio
from Const import *
import CONF
from VisualModels.BLIP import blip_analyzer
from VisualModels.Hiera import video_recognizer
# from Utils.FrameBuffer import FrameBuffer
from Utils.FrameBuffer import frame_buffer
import ActionGeneration as ag


context = zmq.asyncio.Context()

visual_tasks = set()
resp_sending_tasks = set()
# frame_buffer = FrameBuffer()  # TODO add a wrapper may be better to replace this global var


def on_vqa_task(*args):
	if CONF.debug:
		ts = args[-1]
		print(f'on vqa task timestamp: {ts}')
	return blip_analyzer.on_vqa_task(*args[:2])


def on_video_rec_task(*args):
	if CONF.debug:
		ts = args[-1]
		print(f'on video rec task timestamp: {ts}')
	return video_recognizer.on_video_recognition_task(frame_buffer)


def on_video_rec_pose_gen_task(*args):
	if CONF.debug:
		ts = args[-1]
		print(f'on video recognition with pose generation task {ts}')
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

TASK_DISPATCHER = {
	VisualTasks.VQA: on_vqa_task,
	VisualTasks.VideoRecognition: on_video_rec_task,
	VisualTasks.FaceRecognition: on_face_rec_task,
	VisualTasks.VideoRecogPoseGen: on_video_rec_pose_gen_task,
}


def run_background_visual_task(msg, full_cb):

	async def run_task(msg, callback):  # msg: [task_type, img_bytes, xx ]
		encoding = CONF.encoding
		task_type = msg[0].decode(encoding)
		frame_buffer.append_content(msg[1])
		response = TASK_DISPATCHER[task_type](*msg[1:])
		print(f'on task finish response : {type(response), response}')
		if response is None:
			return
		params = [resp.encode(encoding) for resp in response] if isinstance(response, list) else [response.encode(encoding)]
		# if CONF.debug:
		# 	params.append(msg[-1])
		# params = (response.encode(encoding), msg[-1]) if CONF.debug else (response, )
		callback(*params)
		await asyncio.sleep(10.)

	loop = asyncio.get_event_loop()

	coroutine = loop.create_task(
		run_task(msg, full_cb)
	)
	visual_tasks.add(coroutine)
	coroutine.add_done_callback(lambda _:visual_tasks.remove(coroutine))
	# print(f'len of visual tasks: {len(visual_tasks)}')


# ==============================
# TODO should organize these sockets in a tidy way
resp_socket = context.socket(zmq.PUB)
resp_socket.bind('tcp://127.0.0.1:20001')


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
	socket.connect('tcp://127.0.0.1:2000')
	socket.setsockopt(zmq.SUBSCRIBE, b'')
	while True:
		msg = await socket.recv_multipart()
		run_background_visual_task(msg, on_task_finish_cb)


# updated pure vcap data subscription
def run_sub_sync():
	ctx = zmq.Context()
	socket = ctx.socket(zmq.SUB)
	socket.connect('tcp://127.0.0.1:2000')
	socket.setsockopt(zmq.SUBSCRIBE, b'')
	while True:
		msg = socket.recv_multipart()
		img_bytes, ts = msg
		frame_buffer.append_content(img_bytes)
		print(f'subscribe ts: {ts}')


if __name__ == "__main__":
	# arg_test(3, 'name')
	# pass
	asyncio.run(run_sub())
