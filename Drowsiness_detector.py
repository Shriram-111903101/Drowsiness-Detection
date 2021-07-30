# Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for face landmark detection
import dlib
# face_utils for basic operations of conversion
from imutils import face_utils
from pygame import mixer

# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

mixer.init()
sound = mixer.Sound('Alarm1.mp3')

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()

#load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# status marking for current state(flags)
sleep = 0
drowsy = 0
active = 0
status = ""
color = (255, 255, 255)   #white


#calculating the euclidean distace
def compute(pt_a, pt_b):
    dist = np.linalg.norm(pt_a - pt_b)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)       #eye aspect ratio(EAR)

    # Checking if it is blinked
    if (ratio > 0.25):          #active
        return 2
    elif (ratio > 0.21 and ratio <= 0.25):      #drowsy
        return 1
    else:                           #sleepy
        return 0


while True:
    _, frame = cap.read()

    #converting colour image to gray cause the detector only works with grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    # detected face in faces array
    for face in faces:
        #creating our region of interest

        x1 = face.left()    #left point
        y1 = face.top()     #top point
        x2 = face.right()   #right point
        y2 = face.bottom()  #bottom point

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Now judge what to do for the eye blinks
        if (left_blink == 0 or right_blink == 0):
            sleep += 1
            drowsy = 0
            active = 0
            if (sleep > 20):
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                sound.play()
            else:
                sound.stop()


        elif (left_blink == 1 or right_blink == 1):
            sleep = 0
            active = 0
            drowsy += 1
            if (drowsy > 6):
                status = "Drowsy !"
                color = (0, 0, 255)

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if (active > 6):
                status = "Active :)"
                color = (0, 255, 0)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
