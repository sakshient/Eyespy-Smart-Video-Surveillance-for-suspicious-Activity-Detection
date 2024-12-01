# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 

import numpy as np
import argparse
import imutils
import cv2
from imutils.video import VideoStream
import time

from imutils.video import FPS

from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import subprocess
#import lcd
import json

import cv2
import numpy as np
import random
import os

import os
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Gun"]
detresult="None"

output_layer_names = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))


def inFrame(lst):
	if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
		return True 
	return False

model  = load_model("model.h5")
label = np.load("labels.npy")



holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils





cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# creating canvas to draw on it
canvas = np.zeros((720,1280,3), np.uint8)


lss1=subprocess.getoutput('hostname -I')
lss1=lss1.strip()
lss1=lss1.split()
lss=lss1[0]
print(lss)

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)


time.sleep(2.0)




@app.route("/",methods=['GET', 'POST'])
def index():
    # return the rendered template
    
    print(df1)
    return render_template("index.html")

def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global cap, outputFrame, lock, detresult,df1
   

   
    total = 0

    # loop over frames from the video stream

    while True:
        
        #print(coolingCounter)
        
        lst = []

        _, test_img = cap.read()

        window = np.zeros((940,940,3), dtype="uint8")

        test_img = cv2.flip(test_img, 1)

        res = holis.process(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))

        test_img = cv2.blur(test_img, (4,4))
        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1,-1)

            p = model.predict(lst)
            pred = label[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                cv2.putText(window, pred , (180,180),cv2.FONT_ITALIC, 1.3, (0,255,0),2)
                
                blob = cv2.dnn.blobFromImage(test_img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outs = net.forward(output_layer_names)

            # Showing information on the screen
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            print(indexes)
            if indexes == 0: print("weapon detected in frame")
            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(test_img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                    detresult= pred+","+label
                    df1=music_rec()
                    
                
                
                
        

            

            
        drawing.draw_landmarks(test_img, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                connection_drawing_spec=drawing.DrawingSpec(color=(255,255,255), thickness=6 ),
                                 landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), circle_radius=3, thickness=3))

                  
   
        total += 1

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = test_img.copy()
            #outputFrame =cv2.flip(outputFrame, 1)
        
def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock,df1

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

# @app.route('/t')
# def gen_table():
#     return df1

def music_rec():
	df = "{\"suspiscious\":\""+detresult+"\"}"
	print("the ress",df)
	return df


df1 = music_rec()

@app.route('/t')
def gen_table():
    
    print(df1)
    return df1

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
 

    # start a thread that will perform motion detection
    t = threading.Thread(target=detect_motion, args=(
        32,))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=lss, port="8000", debug=True,
        threaded=True, use_reloader=False)

# release the video stream pointer
cap.release()
cv2.destroyAllWindows()

