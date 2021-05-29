# Code that tracks an object that is in a predifined box in the videostream, user can press 'enter' to start the tracker
# and can control the size of the predifined box by (8,6,4,2) on keyboard

# by Muhammad Khan and Haritha Seddik

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse  # for functions
import imutils  # for image processing functions
import time
import cv2  # computer vision and machine learning

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
                help="OpenCV object tracker type")
args = vars(ap.parse_args())


# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None
flag = False
a = 10
b = 230
t_w = 100
t_h = 100

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None

# loop over frames from the video stream
while True:

    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    frame = cv2.flip(frame, 1)
    # time.sleep(0.03)

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=700)
    (H, W) = frame.shape[:2]


# check to see if we are currently tracking an object
    if initBB is not None:
        flag = True
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # update the FPS counter
        fps.update()
        fps.stop()

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    # print(int((frame.shape[0]/2)-(t_h/2)))
    if flag == False:
        cv2.rectangle(frame, (a, b), (a+t_w, b+t_h), (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)  # & 0xFF

    if key == ord(" "):
        initBB = (a, b, t_w, t_h)
        tracker.init(frame, initBB)
        fps = FPS().start()
    elif int(key) == 52:
        a = a + 2
        t_w = t_w - 4
    elif int(key) == 53:
        b = b + 2
        t_h = t_h - 4
    elif int(key) == 54:
        a = a - 2
        t_w = t_w + 4
    elif int(key) == 56:
        b = b - 2
        t_h = t_h + 4
    elif key == ord("q"):
        break

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    # if key == ord("s"):

    # 	initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    # 	print(initBB)
    # 	tracker.init(frame, initBB)
    # 	fps = FPS().start()

# # if the `q` key was pressed, break from the loop
    # elif key == ord("q"):
    # 	break


# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
