# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import os.path as osp

ENCODING = 'utf-8'

N_PIXEL = 1280 * 720 * 3    # num of pixels per frame (Ameca)
IMG_SIZE = (1280, 720)

ATTN_MASK_FILL = -1e38 # -1e-9  #

FPS = 25 
# FRAME_INTERVAL = 1/25

ORIGINAL_IMG_SHAPE = (720, 1280, 3)  # (h, w, c)
FACE_IMG_SHAPE = (160, 160)



class CustomTasks:
	VQA = 'VQA'    
	VideoRecognition = 'VideoRecognition'
	FaceRecognition = 'FaceRecognition'
	EmotionImitation = 'EmotionImitation' 


class ResponseCode:
	KeepSilent = b'0' 
	Success = b'1'
	Fail = b'2'


class Emotions:
	Other = -1
	Neutral = 0
	Surprise = 1
	Fear = 2
	Sadness = 3
	Joy = 4
	Disgust = 5
	Anger = 6


# class Emotions:
# 	Other = -1
# 	Neutral = 0
# 	Angry = 1
# 	Confused = 2
# 	Dislike = 3
# 	Fear = 4
# 	Happy = 5
# 	Sad = 6
# 	Scared = 7
# 	Surprised = 8


EMOTION_TO_ANIM = {
	Emotions.Other: ['Chat Expressions.dir/Chat_G2_Neutral.project',],
	Emotions.Neutral: ['Chat Expressions.dir/Chat_G2_Neutral.project',],
	Emotions.Surprise: [
		'Chat Expressions.dir/Chat_G2_Surprised_1.project',
		'Chat Expressions.dir/Chat_G2_Surprised_2.project',
		'Chat Expressions.dir/Chat_G2_Surprised_3.project',
		],
	Emotions.Fear: [
		'Chat Expressions.dir/Chat_G2_Fear_1.project',
		'Chat Expressions.dir/Chat_G2_Fear_2.project',
		'Chat Expressions.dir/Chat_G2_Fear_3.project'
		],
	
	Emotions.Sadness: [
		'Chat Expressions.dir/Chat_G2_Sad_1.project',
		'Chat Expressions.dir/Chat_G2_Sad_2.project',
		'Chat Expressions.dir/Chat_G2_Sad_3.project',
		],
	
	Emotions.Joy: [
		'Chat Expressions.dir/Chat_G2_Happy_1.project',
		'Chat Expressions.dir/Chat_G2_Happy_2.project',
		'Chat Expressions.dir/Chat_G2_Happy_3.project',
		# 'Chat Expressions.dir/Chat_G2_Happy_with_audio.project',
		],
	
	Emotions.Disgust: [
		'Chat Expressions.dir/Chat_G2_Dislike_1.project',
		'Chat Expressions.dir/Chat_G2_Dislike_2.project',
		'Chat Expressions.dir/Chat_G2_Dislike_3.project'
		],

	Emotions.Anger: [
		'Chat Expressions.dir/Chat_G2_Angry_1.project',
		'Chat Expressions.dir/Chat_G2_Angry_2.project',
		'Chat Expressions.dir/Chat_G2_Angry_3.project',],

}
EMOTION_TO_ANIM = {emo: [osp.join('Animations.dir/System.dir', anim) for anim in anim_list] for emo, anim_list in EMOTION_TO_ANIM.items()}


