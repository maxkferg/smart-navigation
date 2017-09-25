"""
Track an object using a webcam and push the position to redis

Usage: python3 track.py
"""

import cv2
import sys
import json
import time
import redis
from config import *


db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)



if __name__ == '__main__' :

    # Set up tracker.
    # Instead of MIL, you can also use
    # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
    tracker = cv2.bgsegm.createBackgroundSubtractorMOG(200,5,0.90)

    # Read video
    video = cv2.VideoCapture(CAMERA_IP)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Create a video capture window
    cv2.namedWindow("Tracking")

    # Define an initial bounding box
    #bbox = cv2.selectROI("Tracking", frame)

    # Initialize tracker with first frame and bounding box
    #ok = tracker.init(frame, bbox)

    count = 0
    start = time.time()


    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        fgmask = tracker.apply(frame, None, 0.8)
        dilation = cv2.dilate(fgmask, None, iterations=3)
        # erosion = cv2.erode(fgmask,None,iterations = 1)

        im2, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img,contours,-1,(0,255,0),cv2.cv.CV_FILLED,32)


        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []
        for contour, hier in zip(contours, hierarchy):
            (x,y,w,h) = cv2.boundingRect(contour)

            if w > 6 and h > 8 and y < - 200/125*x + 240/125*170 + 200:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (180,0,0), 2)
                x1=w/2
                y1=h/2
                cx=int(x+x1)
                cy=int(y+y1)

                cv2.circle(frame, (cx, cy), 1, (0, 255, 255), -1)
                #a.append([cx,cy])

                # rect = cv2.minAreaRect(contour)
                # box = cv2.cv.BoxPoints(rect)
                # box = np.int0(box)
                # cv2.drawContours(img,[box],0,(0,0,255),2)

        # cv2.putText(img,"Hello World!!!", (0,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        #cv2.imshow('BGS', fgmask)
        #cv2.imshow('Ori+Bounding Box',frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

        count += 1
        print("FPS:",count/(time.time()-start))
        continue



        #if ok:
        #    db.set('position',json.dumps(bbox))

        # Draw bounding box
        if ok and count%10==0:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255))

            # Display result
            cv2.imshow("Tracking", frame)

        count += 1


