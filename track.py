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
    video = cv2.VideoCapture('/home/hussam/Downloads/hd-100.mkv')
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
        #cropped=frame[bbox[1]:bbox[1]+bbox[3],bbox[2]:bbox[2]+bbox[4]]
        # Draw bounding box
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 0, 255))
        end = time.time()
        cropped = frame[p1[1]:p2[1], p1[0]:p2[0]]
        coordinates=np.argwhere(cropped)
        #Getting segmentation of the two circles and getting coordinates of each red and green point
        coordgreen=np.argwhere((cropped[:,:,1]>cropped[:,:,2]*1.15) & (cropped[:,:,1]>cropped[:,:,0]*1.15))
        coordred=np.argwhere((cropped[:,:,2]>cropped[:,:,1]*1.15) & (cropped[:,:,0]>cropped[:,:,1]*1.15))
        loc=np.argmax(coordgreen[:,1])
        loc1=np.argmax(coordred[:,1])
        red=coordred[loc1]
        green=coordgreen[loc]
        #Rotation of robot in radians
        angle=math.atan2(red[0]-green[0],red[1]-green[1])
        #Rotation of robot in Degrees

        angle=math.degrees(angle)
        angletext=str('angle='+str(angle))
      #Center Point of the robot
        centerofCar=[p1[0]+bbox[2]/2,p1[1]+bbox[3]/2]

        cv2.putText(frame,angletext, (5, 60), cv2.FONT_HERSHEY_DUPLEX, 2, 255)
        cropped[coordgreen[:, 0], coordgreen[:, 1]] = 0
        cropped[coordred[:, 0], coordred[:, 1]] = 0

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
        cv2.imshow("cropped",cropped)
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break