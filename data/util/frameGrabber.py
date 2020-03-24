import cv2
import datetime
import numpy as np
import os

cap = cv2.VideoCapture('The Iron Bowl 2014 - #1 Alabama vs. #15 Auburn (Highlights).mp4')

shouldSave = False
saveMod = 10 # save every tenth frame
frameCounter = 0
saveCounter = 1

start = datetime.datetime.now().replace(microsecond=0).isoformat()
start = start.replace(":", ".")
os.mkdir(start)

while(cap.isOpened()):
    ret, frame = cap.read()
    frameCounter += 1

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if (shouldSave and (frameCounter%saveMod == 0)):
        cv2.imwrite(os.path.join(start, str(saveCounter)+".jpg"), frame)
        saveCounter += 1

    cv2.imshow('frame', frame)

    key = cv2.waitKey(2)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        shouldSave = (not shouldSave)
        if shouldSave:
            print("Save ON")
        else:
            saveCounter -= 1 # overwrite the previously saved file
            print("Save OFF")
    elif key & 0xFF == ord('p'):
        print("paused! press any key to continue")
        cv2.waitKey()

cap.release()
cv2.destroyAllWindows()