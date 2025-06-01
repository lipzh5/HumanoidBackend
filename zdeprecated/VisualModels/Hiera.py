# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: Hiera for action recognition
import base64
import io
from io import BytesIO

import CONF
import Const
from Utils import DataUtils
import hiera
from Utils.FrameBuffer import FrameBuffer
from Const import *
from CONF import HieraConf
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import random

CHECK_POINT = 'mae_k400_ft_k400'

id_to_name_map = DataUtils.get_gt_labels_for_k400()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class VideoRecognizer:
	def __init__(self):
		self.model = hiera.hiera_base_16x224(pretrained=True, checkpoint=CHECK_POINT).to(device).eval()

	@staticmethod
	def get_processed_frames(frame_buffer):
		# use the newest frames
		frames = np.stack([
			np.asarray(Image.open(io.BytesIO(base64.b64decode(frame_buffer.buffer_content[-i-1]))))
			for i in range(HieraConf.n_frames_per_video)
		]) if CONF.debug else np.stack([
			np.asarray(Image.open(BytesIO(frame_buffer.buffer_content[-i-1])))
			# np.asarray(Image.frombytes('RGB', IMG_SIZE, frame_buffer.buffer_content[-i-1]))
			for i in range(HieraConf.n_frames_per_video)])

		frames = torch.tensor(frames).float() / 255
		frames = torch.stack([frames[i * HieraConf.n_frames_per_clip: (i + 1) * HieraConf.n_frames_per_clip] for i in
							  range(HieraConf.n_infer_clips)])
		frames = frames.permute(0, 4, 1, 2, 3).contiguous()
		print(f'action recognition frames shape: {frames.shape}')
		frames = F.interpolate(frames, size=HieraConf.input_size, mode=HieraConf.input_interp_mode)
		# Normalize the clips TODO mean and std??
		frames = frames - torch.tensor([0.45, 0.45, 0.45]).view(1, -1, 1, 1, 1)  # subtract mean
		frames = frames / torch.tensor([0.225, 0.225, 0.225]).view(1, -1, 1, 1, 1)  # divide by std
		return frames.to(device)

	def recognize_action(self, frame_buffer):
		frames = self.get_processed_frames(frame_buffer)
		out = self.model(frames)
		out = out.mean(0)
		out = out.argmax(dim=-1).item()
		return out  # label of the action
		# return id_to_name_map[out]

	def on_video_recognition_task(self, frame_buffer):
		# FrameBuffer.append_content(frame)
		print(f' video recog frame buffer len: {len(frame_buffer.buffer_content)}')
		if len(frame_buffer.buffer_content) < HieraConf.n_frames_per_video:
			return None  # at least on clip for inference
		action_label = self.recognize_action(frame_buffer)
		return id_to_name_map[action_label]

	def on_video_rec_posegen_task(self, frame_buffer):
		"""return human_action_name and emotion_anim_project"""
		print(f' video recog frame buffer len: {len(frame_buffer.buffer_content)}')
		if len(frame_buffer.buffer_content) < HieraConf.n_frames_per_video:
			return None  # at least on clip for inference
		action_label = self.recognize_action(frame_buffer)
		emotion_label = DataUtils.emotion_labels[action_label]
		return id_to_name_map[action_label], random.choice(EMOTION_TO_ANIM[emotion_label])



video_recognizer = VideoRecognizer()


# def try_recognize_action(frame: bytes):
# 	FrameBuffer.append_content(frame)
# 	if len(FrameBuffer.buffer_content) < HieraConf.n_frames_per_video:
# 		return None   # at least on clip for inference
# 	return recognize_action()

# def recognize_action():
# 	frames = np.stack([
# 		np.asarray(Image.frombytes('RGB', IMG_SIZE, FrameBuffer.buffer_content.popleft()))
# 		for _ in range(HieraConf.n_frames_per_clip)])
# 	frames = torch.tensor(frames).float() / 255
# 	frames = torch.stack([frames[i*HieraConf.n_frames_per_clip: (i+1)*HieraConf.n_frames_per_clip] for i in range(HieraConf.n_infer_clips)])
# 	frames = frames.permute(0, 4, 1, 2, 3).contiguous()
# 	print(f'action recognition frames shape: {frames.shape}')
# 	frames = F.interpolate(frames, size=HieraConf.input_size, mode=HieraConf.input_interp_mode)
# 	# Normalize the clips TODO mean and std??
# 	frames = frames - torch.tensor([0.45, 0.45, 0.45]).view(1, -1, 1, 1, 1)  # subtract mean
# 	frames = frames / torch.tensor([0.225, 0.225, 0.225]).view(1, -1, 1, 1, 1)  # divide by std
#
# 	# TODO human action recognition and then robot pose generation
# 	pass


if __name__ == "__main__":
	t = list(range(20))
	step = 2
	l = [t[i:i+1] for i in range(5)]


