#!/usr/bin/env python3

#  import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import imutils
import time
import cv2
import os
import ImageSize
from Reward import Reward
import datetime
from Motor import ServoMotor
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--faces", required=False, const=True , default=False, nargs='?',
                help="should we hide faces ")
ap.add_argument("-g", "--grayscale", required=False, const=True, default=False,nargs='?',
                help="should we use gray scale ")
ap.add_argument("-r", "--revert", required=False, const=True, default = False,nargs='?',
                help="Should we revert the image")
args = vars(ap.parse_args())

if args["faces"] :
    print("[INFO] will be identifying faces")
if args["grayscale"] :
    print("[INFO] will be using grayscale")
if args["revert"] :
    print("[INFO] will be flipping verticaly the frames")


# define the paths to the Not Hookie Keras deep learning model and
# audio file
MODEL_PATH = "hookie.model"

path = './haarcascade_frontalface_default.xml'
if(not os.path.exists(path)):
    raise Exception("haar topology file not found")



# initialize the total number of frames that *consecutively* contain
# hookie along with threshold required to trigger the santa alarm
TOTAL_CONSEC = 0
TOTAL_THRESH = 20
LAST_SEEN = datetime.datetime.now()

# initialize is the Hookie alarm has been triggered
HOOKIE_FOUND = False

def init_motor():
    print("[INFO] motor initialization...")
    motor = ServoMotor()
    motor.setup()
    try:
        for i in range(0,180):
            motor.Rotate(0)
    finally :
        motor.destroy()
        
def rotate_motor():
    motor = ServoMotor()
    motor.setup()
    try:
        motor.loopOnce()
    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the child program destroy() will be  executed.
        motor.destroy()
    finally:
        motor.destroy()

def ActionateMotor(LAST_SEEN,sync=True):
    elaspsed = datetime.datetime.now() - LAST_SEEN
    if elaspsed.seconds > 3 : #trigger the motor
        LAST_SEEN = datetime.datetime.now()
        if sync :
            rotate_motor()
        else :
            treeThread = Thread(target=rotate_motor, args=())
            treeThread.daemon = True
            treeThread.start()

def PlaySentence(TTS,LAST_SEEN,sync=True):
    elaspsed = datetime.datetime.now() - LAST_SEEN
    if elaspsed.seconds > 3:
        LAST_SEEN = datetime.datetime.now()
        if sync :
            TTS.sentence()
        else :
            LAST_SEEN = datetime.datetime.now()
            talkThread = Thread(target=TTS.sentence, args=())
            talkThread.daemon = False
            talkThread.start()


def ProcessFrame(frame):
    frame = imutils.resize(frame, width=400)
    # prepare the image to be classified by our deep learning network
    image = cv2.resize(frame, (ImageSize.HEIGHT, ImageSize.WIDTH))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image



if __name__ == '__main__':

   motor = ServoMotor()
   motor.setup()
   try:
        TTS = Reward()
        motor.loopOnce()

        # load the model
        print("[INFO] loading model...")
        model = load_model(MODEL_PATH)

        # initialize the video stream and allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()

        time.sleep(2.0)

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = vs.read()
            image = ProcessFrame(frame)

            # classify the input image and initialize the label and
            # probability of the prediction
            (notHookie, hookie) = model.predict(image)[0]
            label = "Not Hookie"
            proba = notHookie

            # check to see if Hookie was detected using our convolutional neural network
            if hookie > notHookie:

                # update the label and prediction probability
                label = "Hookie"
                proba = hookie

                # increment the number of frame with Hookie
                TOTAL_CONSEC += 1

                # check to see if we should raise the Hookie alarm
                if not HOOKIE_FOUND and TOTAL_CONSEC >= TOTAL_THRESH:

                    HOOKIE_FOUND = True # indicate that Hookie has been found
                    #motor.loopOnce()
                    ActionateMotor(LAST_SEEN, True)
                    PlaySentence(TTS,LAST_SEEN, False)
                    

            # otherwise, reset the  number of consecutive frames
            else:
                TOTAL_CONSEC = 0
                HOOKIE_FOUND = False

            if args["revert"] :
                frame = cv2.flip(frame,1)
            
            # build the label and draw it on the frame
            label = "{}: {:.2f}%".format(label, proba * 100)
            frame = cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            
            if args["faces"] :
                face_cascade = cv2.CascadeClassifier(path)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            if args["grayscale"] :
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # show the output frame
            cv2.imshow("Hookie screen", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
   finally :
        # do a bit of cleanup
        print("[INFO] cleaning up...")
        cv2.destroyAllWindows()
        vs.stop()
        motor.destroy()
        
        
