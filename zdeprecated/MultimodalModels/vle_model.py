# -*- coding:utf-8 -*-
# @Author: Peizhen Li 
# @Desc: None

import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import random
from transformers import AutoModel
from MultimodalModels.Modules.transformer import TransformerEncoder, AdditiveAttention
from MultimodalModels.Modules.cross_modal_transformer import CrossModalTransformerEncoder
from VisualModels.EmotionRec import emo_recognizer
# from models.model_kd_teacher import KDTeacher
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
log = logging.getLogger(__name__)

# from utils import *
from MultimodalModels.multimodal_utils import *
from mmcv.cnn import xavier_init

from Const import ATTN_MASK_FILL, FPS, Emotions,EMOTION_TO_ANIM, ORIGINAL_IMG_SHAPE, FPS
from CONF import MAX_FACES, MAX_FRAMES
import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image
import io
import base64
import os



print(f'***** \n multimodal torch seed: {torch.initial_seed()}')


class VLEModel(nn.Module):
	"""vision language to emotion"""
	def __init__(self, cfg):
		print(f'***** \n 777multimodal init torch seed: {torch.initial_seed()}')
		print(f'***** \n 777multimodal init cuda seed: {torch.cuda.initial_seed()}')
		super().__init__()
		# cross transformer modules
		transformer_conf = cfg.model.transformers
		hidden_size = transformer_conf.hidden_size
		tenc_cfg = cfg.model.text_encoder
		self.text_linear = nn.Linear(tenc_cfg.embed_dim, hidden_size)   
		self.vision_linear = nn.Linear(cfg.data.vision_feature_dim, hidden_size)
		"""vision"""
		self.vision_utt_transformer = TransformerEncoder(
		transformer_conf.self_attn_transformer,
		transformer_conf.self_attn_transformer.num_transformer_layers.vision,
		cfg.data.vision_utt_max_len, hidden_size)

		self.cm_tv_transformer = CrossModalTransformerEncoder(
			hidden_size,
			**transformer_conf.cross_modal_transformer.text_vision)
		self.additive_attn = AdditiveAttention(hidden_size, hidden_size)
		# self.multimodal_projection = nn.Linear(100+256, 1) 
		# print(f'no additive attention, use linear projection instead!!!\n******')

		self.dropout = nn.Dropout(transformer_conf.self_attn_transformer.hidden_dropout_prob)
		self.classifier = nn.Linear(hidden_size, cfg.data.num_labels)

		self._init_weights()
		
		"""text"""
		self.context_encoder = AutoModel.from_pretrained(tenc_cfg.pretrained_path, local_files_only=False)
		self.pad_value = tenc_cfg.pad_value   
		self.mask_value = tenc_cfg.mask_value 

		# self.vfeat_from_pkl = cfg.train.vfeat_from_pkl
		# if not cfg.train.vfeat_from_pkl:
		self.vision_encoder = set_vision_encoder(cfg)

		
	def _init_weights(self):
		# ref to: https://github.com/junjie18/CMT/tree/master
		for m in self.modules():
			if hasattr(m, 'weight') and m.weight.dim() > 1:
				xavier_init(m, distribution='uniform')
		self._is_init = True

	def gen_text_reps(self, sentences, text_mask):
		"""generate vector representation for each turn of conversation"""
		batch_size, max_len = sentences.shape[0], sentences.shape[-1]
		sentences = sentences.reshape(-1, max_len)
		# mask = 1 - (sentences == (self.pad_value)).long()
		utterance_encoded = self.context_encoder(
			input_ids=sentences,
			attention_mask=text_mask,
			output_hidden_states=True,
			return_dict=True
		)['last_hidden_state']
		return self.text_linear(utterance_encoded)  # NOTE: Different from SPCL paper, we use all token reps!!!!

	
	def gen_vision_reps(self, img_inputs, vision_mask): 
		bs, max_utt_img_len, channel, width, height = img_inputs.shape
		img_inputs = img_inputs.reshape(bs*max_utt_img_len, channel, width, height)
		vision_mask = vision_mask.reshape(bs*max_utt_img_len)
		real_img_inputs = img_inputs[vision_mask>0]

		embeddings = self.vision_encoder(real_img_inputs)
		embedding_dim = embeddings.shape[-1]
		output_embeddings = torch.zeros((bs*max_utt_img_len, embedding_dim)).to(img_inputs.device)
		output_embeddings[vision_mask>0] = embeddings
		output_embeddings = output_embeddings.reshape(bs, max_utt_img_len, -1)
		return self.vision_linear(output_embeddings)
	

	
	def forward(self, text_input_ids, vision_inputs, vision_mask):

		text_mask = 1 - (text_input_ids == (self.pad_value)).long()
		text_utt_linear = self.gen_text_reps(text_input_ids, text_mask).transpose(1, 0)  # [256, bs, 768]
		
		vision_linear = self.gen_vision_reps(vision_inputs, vision_mask)

		vision_extended_utt_mask = vision_mask.unsqueeze(1).unsqueeze(2)
		vision_extended_utt_mask = (1.0 - vision_extended_utt_mask) * ATTN_MASK_FILL

		vision_utt_trans = self.vision_utt_transformer(vision_linear, vision_extended_utt_mask).transpose(1, 0) 
		# text cross vision
		text_vision_attn = self.cm_tv_transformer(text_utt_linear, vision_utt_trans, vision_utt_trans)
		vision_text_attn = self.cm_tv_transformer(vision_utt_trans, text_utt_linear, text_utt_linear)
		text_vision_cross_feat = torch.concat((text_vision_attn, vision_text_attn), dim=0)
		text_vision_utt_mask = torch.concat((text_mask, vision_mask), dim=1)  # (1, 256), (1, 130)
	
		# multimodal_out = self.multimodal_projection(text_vision_cross_feat.permute(1, 2, 0)).squeeze(-1)  # (seq_len, bs, emb_dim)
		multimodal_out, _ = self.additive_attn(text_vision_cross_feat.transpose(1,0), text_vision_utt_mask)
		return self.classifier(self.dropout(multimodal_out))


def get_model_instance(cfg) -> None:
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)
	random.seed(cfg.seed)
	
	model = VLEModel(cfg)
	model.eval()
	if osp.exists(cfg.train.save_model_path):
		print(f'load state dict: {cfg.train.save_model_path} \n ******')
		state_dict = torch.load(cfg.train.save_model_path, map_location=torch.device('cuda'))
		model.load_state_dict(state_dict['model'])
	model.requires_grad_(False)   # frozen teacher
	return model

cfg = OmegaConf.load('./MultimodalModels/model_conf.yaml')
model = get_model_instance(cfg).cuda()
print(f'model type in vle model: {type(model)}')


async def get_emotion_response(ts_end, duration):
	text_input_ids = get_text_inputs_from_raw()
	n_frames = min(int(duration * FPS), MAX_FRAMES)  
	
	img_inputs, img_mask = get_vision_inputs_from_raw(n_frames)
	logits = model(text_input_ids.unsqueeze(0).cuda(), img_inputs.unsqueeze(0).cuda(), img_mask.unsqueeze(0).cuda())
	emotion_label = torch.argmax(logits, dim=-1).item()
	if emotion_label == Emotions.Neutral:
		# query GPT using visual observation
		anim = await emo_recognizer.on_emotion_recog_for_vle(frame_buffer.buffer_content[-1])
		print(f'anim from gpt: {anim} \n*****')
		return anim
	print(f'get emotion response logits: {logits} \n ******************')
	emotion_list = EMOTION_TO_ANIM.get(emotion_label, [])
	anim = '' if not emotion_list else random.choice(emotion_list)

	return anim


class VLETaskDealer:
	def __init__(self):
		super().__init__()
		self.ameca_just_said = False
		self.ameca_sentence = b''   # Ameca may send sentence in several pieces
	
	async def on_vle_task(self, utterance, ts_end, duration, from_ameca, diag_buffer):
		if utterance == b'ameca starts new dialogue':
			diag_buffer.clear_buffer()
			print(f'starts new dialogue!!!\n*****')
			return ResponseCode.KeepSilent, ''
		
		if int(from_ameca.decode(encoding)) > 0:  # sentence from Ameca
			self.ameca_just_said = True
			self.ameca_sentence += (b' ' + utterance)
			return ResponseCode.KeepSilent, ''
		else: 
			if self.ameca_just_said:
				print(f'ameca sentence: {self.ameca_sentence} \n*****')
				self.ameca_just_said = False   # human starts to respond to ameca
				self.ameca_sentence and diag_buffer.update_dialogue(self.ameca_sentence)
				self.ameca_sentence = b''       # add cached ameca sentence as well as current human dialogue
				diag_buffer.update_dialogue(utterance) 
			else:
				diag_buffer.update_dialogue(utterance)
		
		emo_anim = await get_emotion_response(float(ts_end.decode(encoding)), float(duration.decode(encoding)))
	
		return ResponseCode.Success, emo_anim 

vle_task_dealer = VLETaskDealer()
	






		



