import cv2
import sys
import time
import numpy as np
import math

if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use
    # BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN

    tracker = cv2.Tracker_create("KCF")

    # Read video
    video = cv2.VideoCapture('/path/to/video')

    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot read video file'
        sys.exit()

    # Define an initial bounding box
   # bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)
    total=1
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    totalseconds=0
    while True:
        # Read a new frame
        start = time.time()
        ok, frame = video.read()
        if not ok:
            break

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Draw bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 0, 255))
        end = time.time()

        # Time elapsed
        seconds = end - start
        totalseconds = totalseconds + seconds
        total=total+1
        if total/30 == float(total/30.0):
            # Calculate frames per second
            fps = float(30/totalseconds)
            totalseconds=0
            print "Estimated frames per second : {0}".format(fps)
        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break