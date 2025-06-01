# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None

import insightface
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from PIL import Image
from numpy import asarray
import os
from io import BytesIO
import time
import os.path as osp
cwd = osp.abspath(osp.dirname(__name__))
import sys
sys.path.append(cwd)
from Const import *
# from collections import deque

app = FaceAnalysis(providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])  # 'CUDAExecutionProvider'
app.prepare(ctx_id=0, det_size=(640, 640))


FACE_DB_PATH = os.path.join(os.path.dirname(__file__), '../Assets/face_db')
COS_SIM_THRESHOLD = 0.5
SAY_HELLO_INTERVAL = 60 # 3 * 3600  # 3 HOURS


class FaceEmbdCacheObj:
	embed_dim = 512  # embedding dim
	embed_cache = None  # np.ndarray, normalized embeddings
	name_cache = None   # a list of names
	name_tstamp = None  # {name: timestamp of last meet}

	@classmethod
	def fetch_from_cache(cls, embed):
		"""
		:param embed: face embeddings in the query image(current view)
		:return: name of the person fetched from storage with highest similarity
		"""
		if not cls.embed_cache or not cls.name_cache:
			return None
		query_cnt = embed.shape[0]  # num of face embds to query
		cos_sims = cls.embed_cache @ np.transpose(embed)
		matched_idx = np.argmax(cos_sims) // query_cnt
		return cls.name_cache[matched_idx]

	@classmethod
	def reset(cls):
		# update cache from face_db
		print('reset cache obj path exists? ', os.path.exists(FACE_DB_PATH))
		embeddings = []
		cls.name_cache = []
		for (dirpath, dirnames, filenames) in os.walk(FACE_DB_PATH):
			print(f'dirpath: {dirpath}, \n dirnames: {dirnames}')
			print('----------')
			print(f'filenames: {filenames}')
			print('====================')
			for filename in filenames:
				image = Image.open(os.path.join(dirpath, filename))
				faces = app.get(asarray(image.convert("RGB")))  # assume one face per image in the face db
			
				if not faces:
					print(f'{filename} does not contain faces \n ************')
					continue
				embeddings.append(faces[0].normed_embedding)
				name = os.path.splitext(filename)[0].split('_')
				# name = filename.splitext()
				# print(f'stripped name: {name}')
				# name = name[0].split('_')
				try:
					cls.name_cache.append(name[0])
				except Exception as e:
					print(str(e))
					print(f'name error: {filename} \n #####')
		cls.embed_cache = np.array(embeddings)
		cls.name_tstamp = {}


FaceEmbdCacheObj.reset()


def find_from_db(frame:bytes, ignore_ts=1):
	"""if ignore_ts = False, then should check detect interval"""
	# img_arr = np.asarray(Image.frombytes('RGB', IMG_SIZE, frame))
	faces = app.get(np.asarray(Image.open(BytesIO(frame))))
	# faces = app.get(np.asarray(Image.frombytes('RGB', IMG_SIZE, frame)))  # can be many faces in the current image
	# if queried by human on purpose, then ignore_ts = True, otherwise in proactive model
	fail_code = ResponseCode.Fail if ignore_ts>0 else ResponseCode.KeepSilent 
	if not faces:
		return fail_code, None
	embeds = [face.normed_embedding for face in faces]
	embeds = np.array(embeds)
	# print(f'query embeds: {embeds.shape}')
	cached_embeds = FaceEmbdCacheObj.embed_cache

	cos_sim = cached_embeds @ np.transpose(embeds)  # (len_cache, len_query_faces)
	# print('type of cos sim: ', type(cos_sim), cos_sim.shape)
	# print('cos sim: ', cos_sim)
	idx = np.argmax(cos_sim)
	row, col = idx // len(faces), idx % len(faces)
	# print('row: ', row, 'col: ', col)
	if cos_sim[row, col] < COS_SIM_THRESHOLD:  # TODO sim Th
		return fail_code, None
	found = FaceEmbdCacheObj.name_cache[row]
	ts = FaceEmbdCacheObj.name_tstamp.get(found)
	now = time.time()
	if not ignore_ts and ts is not None and now - ts < SAY_HELLO_INTERVAL:
		print(f'found {found} but has said hello {now - ts} ago')
		return fail_code, None
	FaceEmbdCacheObj.name_tstamp[found] = now
	# print('type if idx: ', type(idx), idx.shape)
	print(f'found name: {found}')
	return ResponseCode.Success, FaceEmbdCacheObj.name_cache[row]


if __name__ == "__main__":
	pass
