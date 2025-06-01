# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: BLIP VQA model

import torch.cuda
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import base64
import io
from io import BytesIO
import torch

import CONF
from Const import *

MODEL_ID = 'Salesforce/blip-vqa-capfilt-large'


class BlipImageAnalyzer:
	def __init__(self):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model = BlipForQuestionAnswering.from_pretrained(MODEL_ID).to(self.device).eval()
		self.processor = BlipProcessor.from_pretrained(MODEL_ID)
		print(f'Blip model loaded to {self.device}')

	def generate(self, raw_image: Image, question: str) -> str:
		inputs = self.processor(raw_image, question, return_tensors='pt').to(self.device)
		out = self.model.generate(**inputs, max_new_tokens=100, min_new_tokens=1)
		return self.processor.decode(out[0], skip_special_tokens=True)

	def on_vqa_task(self, img_bytes, query: bytes):
		if CONF.debug:
			decoded = base64.b64decode(img_bytes)  # msg[0] is base64 encoded
			raw_image = Image.open(BytesIO(decoded)).convert('RGB')=
		else:
			raw_image = Image.open(BytesIO(img_bytes))
			# raw_image = Image.frombytes('RGB', IMG_SIZE, img_bytes)
		return self.generate(raw_image, query.decode(encoding=CONF.encoding))


blip_analyzer = BlipImageAnalyzer()



# def on_vqa_task(b64_encoded_img, bytes_query):
# 	decoded = base64.b64decode(b64_encoded_img)  # msg[0] is base64 encoded
# 	raw_image = Image.open(io.BytesIO(decoded)).convert('RGB')
# 	return run_vqa_from_client_query(raw_image, bytes_query.decode('utf-8'))
#
#
# def run_vqa_from_client_query(raw_image, question):
# 	processor = BlipProcessor.from_pretrained('Salesforce/blip-vqa-capfilt-large')
# 	model = BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-capfilt-large')
# 	# raw_image = Image.open(image_path).convert('RGB')
# 	inputs = processor(raw_image, question, return_tensors='pt')
# 	# print(f'type of inputs: {type(inputs)}') # <class 'transformers.image_processing_utils.BatchFeature'>
# 	out = model.generate(**inputs)
# 	decoded_out = processor.decode(out[0], skip_special_tokens=True)
# 	# print('decoded output: ', decoded_out)
# 	return decoded_out


if __name__ == "__main__":
	raw_img = Image.open('../Assets/img/image_phone.png').convert('RGB')
	print(f'type of raw img: {type(raw_img)}')
	run_vqa_from_client_query(raw_img, 'what is this in my hand?')
