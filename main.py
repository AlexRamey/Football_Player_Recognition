# Author: Alex Ramey
# Date: April 12, 2020
# main.py accepts the path to a video clip of college football broadcast footage
# and plays it back with player numbers boxed.
# TODO: Accept Roster File
# TODO: Stream results to stdout
# TODO: Add Game Context with OCR

import cv2
from detector import Detector
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
import os
import sys
import threading
import time

# Shared Thread Memory
plainFrame, labelledFrame = None, None

def playVideo(video, lock):
    global plainFrame, labelledFrame
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while(cap.isOpened()):
        ret, frame = cap.read()
        with lock:
            if (plainFrame is None) and (labelledFrame is None):
                plainFrame = frame.copy() # start-up condition
            elif (plainFrame is None) and (labelledFrame is not None):
                cv2.imshow('frame', labelledFrame)
                plainFrame = frame.copy()
                labelledFrame = None
        key = cv2.waitKey(int(1000/fps + .5))
        if key & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def detector(lock):
    global plainFrame, labelledFrame
    d = Detector()
    while(True):
        time.sleep(.01) # ~100 Hz
        frame = None
        with lock:
            if (plainFrame is not None) and (labelledFrame is None):
                frame = plainFrame       
        if frame is None:
            continue
        
        # run detector on the frame
        results = d.detect(frame)
        frame = Image.fromarray(frame)
        annotateFrame(frame, results)
        with lock:
            plainFrame = None
            labelledFrame = np.array(frame)

def annotateFrame(frame, results):
    colors = list(ImageColor.colormap.values())
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(frame)

    boxes = results['rois']
    for i, box in enumerate(boxes):
        top = box[0]
        bottom = box[2]
        left = box[1]
        right = box[3]
        class_name = results['class_names'][i]
        label = "{}: {}%".format(class_name, int(results['scores'][i] * 100))
        color = colors[int(class_name) % len(colors)]
        segments = [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)]
        draw.line(segments, width=4, fill=color)
        label_width, label_height = font.getsize(label)
        margin = np.ceil(0.05 * label_height)
        if top > (label_height + 2 * margin): # 0.05% margin
            label_bottom = top
        else:
            label_bottom = bottom + label_height + 2 * margin
        draw.rectangle([(left, label_bottom - label_height - 2 * margin), (left + label_width, label_bottom)], fill=color)
        draw.text((left + margin, label_bottom - label_height - margin), label, fill="black", font=font)

def main():
    if ((len(sys.argv) != 2) or (not os.path.exists(sys.argv[1]))):
        sys.exit("Please provide the input video file as a cmd line argument")

    lock = threading.Lock()

    threading.Thread(target=detector, args=(lock,), daemon=True).start()

    playVideo(sys.argv[1], lock)

if __name__ == '__main__':
    main()
