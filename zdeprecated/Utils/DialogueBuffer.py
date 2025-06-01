# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None
import torch
from collections import deque
from CONF import diag_buffer_max_len, target_size, encoding
import time


class DialogueBuffer:
	_instance = None
	dialogue = deque()

	def __new__(cls, *args, **kwargs):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
		return cls._instance
	

	@classmethod
	def update_dialogue(cls, utterance):  # TODO add speaker info later
		while len(cls.dialogue) >= diag_buffer_max_len:
			cls.dialogue.popleft()
		cls.dialogue.append(utterance)
		print(f'dia buffer dialog :{len(cls.dialogue)}, utterance: {utterance} \n ****')
	
	@classmethod
	def clear_buffer(cls):
		cls.dialogue.clear()

	def __len__(self):
		return len(self.dialogue)


diag_buffer = DialogueBuffer()