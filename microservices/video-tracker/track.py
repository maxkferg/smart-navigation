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
    tracker = cv2.TrackerMIL_create()

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
    bbox = cv2.selectROI("Tracking", frame)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    count = 0
    start = time.time()

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Update tracker
        ok, bbox = tracker.update(frame)

        if ok:
            db.set('position',json.dumps(bbox))

        # Draw bounding box
        if ok and count%10==0:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255))

            # Display result
            cv2.imshow("Tracking", frame)

        count += 1
        print("FPS:",count/(time.time()-start))

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

