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
    draw = ImageDraw.Draw(frame)
    colors = list(ImageColor.colormap.values())
    draw.line([(0, 0), (0, 100), (100, 100), (100, 0), (0, 0)], width=4, fill=colors[0])

def main():
    if ((len(sys.argv) != 2) or (not os.path.exists(sys.argv[1]))):
        sys.exit("Please provide the input video file as a cmd line argument")

    lock = threading.Lock()

    threading.Thread(target=detector, args=(lock,), daemon=True).start()

    playVideo(sys.argv[1], lock)

if __name__ == '__main__':
    main()
