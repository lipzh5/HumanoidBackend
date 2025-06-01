import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import sys
import os.path as osp
cur_path = osp.abspath(osp.dirname(__file__))
print(f'cur path: {cur_path} \n****')
sys.path.append(cur_path)
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np
from PIL import Image
import io
import base64

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

FRAMES_TO_DETECT = 100   # detect the last 10 frames
STEP = 4 

class FallDetector:
    def __init__(self):
        super().__init__()
        # self.model = Model()

        weights, save_txt, imgsz, trace = osp.join(cur_path, "yolov7.pt"), False, 640, True

        # Directories
        # save_dir = Path(increment_path(Path("runs/detect") / "exp", exist_ok=False))  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        self.stride = stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, 640)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once= imgsz
        self.model = model

        # # Detect
        # im0s = np.zeros((1000, 800, 3))
        # detect(im0s, model)
    
    def detect(self, frame_buffer):
        for i in range(-FRAMES_TO_DETECT, 0, STEP):
            frame = np.asarray(Image.open(io.BytesIO(frame_buffer.buffer_content[i])))
            fall = self.detect_one_frame(frame)
            if fall:
                return fall
        return False


    def detect_one_frame(self, frame):
        # frames = np.stack([
        #     np.asarray(Image.open(io.BytesIO(frame_buffer.buffer_content[i])))
        #     for i in range(-FRAMES_TO_DETECT, 0, STEP)
        # ])
        # # frames = torch.tensor(frames)
        # frames = np.asarray(Image.open(io.BytesIO(frame_buffer.buffer_content[-1])))
        # print(f'frames: {frames.shape} \n*****')

        img = letterbox(frame, 640, self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # print(f'img shape: {img.shape} \n ******')

        # Inference
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', frame
            # name = "result.jpg"
            # save_path = str(save_dir / name)  # img.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if cls == 0:  # detecting the person
                        weight = xyxy[2] - xyxy[0]
                        height = xyxy[3] - xyxy[1]
                        # print(f'cls is 0: {weight / height} \n *****')
                        fall = (weight / height) > 1  # if weight > height, then it's fall
                        if fall:
                            return fall
        return False 


        

fall_detector = FallDetector()



def detect(im0s, names):
    img = letterbox(im0s, 640, stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        s, im0 = '', im0s

        # name = "result.jpg"
        # save_path = str(save_dir / name)  # img.jpg

        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if cls == 0:  # detecting the person
                    weight = xyxy[2] - xyxy[0]
                    height = xyxy[3] - xyxy[1]
                    fall = (weight / height) > 1  # if weight > height, then it's fall
                    if fall:
                        return fall
    return False 

                    # if save_img:  # Add bbox to image
                    #     if fall:
                    #         label = f'{names[int(cls)]} fall {conf:.2f}'
                    #         plot_one_box(xyxy, im0, label=label, color=[0, 0, 255], line_thickness=1)
                    #     else:
                    #         label = f'{names[int(cls)]} not fall {conf:.2f}'
                            # plot_one_box(xyxy, im0, label=label, color=[0, 255, 0], line_thickness=1)

        # # Save results (image with detections)
        # cv2.imwrite(save_path, im0)
        # print(f" The image with the result is saved in: {save_path}")
    
        





if __name__ == '__main__':
    weights, save_txt, imgsz, trace = "yolov7.pt", False, 640, not False

    # Directories
    save_dir = Path(increment_path(Path("runs/detect") / "exp", exist_ok=False))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device('')

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, 640)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once= imgsz

    # Detect
    im0s = np.zeros((1000, 800, 3))
    fall = detect(im0s, model)
    print(f'fall?? :{fall} \n ****')