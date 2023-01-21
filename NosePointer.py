import warnings
warnings.filterwarnings("ignore")

import joblib
import mediapipe as mp
import cv2 as cv
import numpy as np
import pandas as pd
from datetime import datetime
from screeninfo import get_monitors
import sys
from time import time

from pynput.mouse import Button, Controller # mouse control mac

def move_mouse_x(nose_xy, left_eye_xy, right_eye_xy, X, Y, handler):
    
    nx = nose_xy[0]
    lx = left_eye_xy[0]
    rx = right_eye_xy[0]
    
    distance_eyes = np.abs(lx - rx)

    left_distance = np.abs(nx - lx)
    dist_prop_left = 1 - round(left_distance / distance_eyes,2)

    right_distance = np.abs(nx - rx)
    dist_prop_right = 1 - round(right_distance / distance_eyes,2)
    

    if (dist_prop_left >= 0.6):
        X = X - 2
        handler.position = (X, Y)
    
    if (dist_prop_left >= 0.8):
        X = X - 20
        handler.position = (X, Y)

    if (dist_prop_right >= 0.6):
        X = X + 2
        handler.position = (X, Y)
        
    if (dist_prop_right >= 0.8):
        X = X + 20
        handler.position = (X, Y)
    
    return X, Y
    

def move_mouse_y(nose_xy, left_eye_xy, right_eye_xy, X, Y, handler, y_buffer):
    
    ny = nose_xy[1]
    
    if len(y_buffer) < 10:
        y_buffer.append(ny)
    else:
        y_buffer.insert(0, ny)
        y_buffer = y_buffer[:-1]
    
    avg_y = np.mean(y_buffer)
    #print("avg nose", avg_y)
    
    # average distance 21px
    
    if (ny < (avg_y - 20)):
        Y = Y - 2
        handler.position = (X, Y)
        
    if (ny < (avg_y - 40)):
        Y = Y - 10
        handler.position = (X, Y)
       
    if (ny > (avg_y + 20)):
        Y = Y + 2
        handler.position = (X, Y)
        
    if (ny > (avg_y + 40)):
        Y = Y + 10
        handler.position = (X, Y)
        
    return X, Y


def buffer_set(buffer, value,buffer_size, dynamic = True):
    if len(buffer) < buffer_size:
        buffer.append(value)
    else:
        if dynamic:
            buffer.insert(0, value)
            buffer = buffer[:-1]
    return np.mean(buffer), buffer

# eye box open height
def open_len(arr):
    y_arr = []

    for _,y in arr:
        y_arr.append(y)

    min_y = min(y_arr)
    max_y = max(y_arr)

    return max_y - min_y


def eye_closed(buff_mean, eye_readings, buffer_size):
    
    if (len(eye_readings) >= buffer_size):
        eye_closed = (buff_mean / np.mean(eye_readings)) < 0.65
        return eye_closed
    else:
        return False

    
def mouth_open(up, down, mouth_open_size):
    if np.abs(down-up) > mouth_open_size:
        return True
    else:
        return False

# mouse control
mouse = Controller()
    
# mediapipe facemesh handler 
mp_face_mesh = mp.solutions.face_mesh


# mediapipe facemesh coordinates... 
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
NOSE_CENTER = [4]
LIP_UP = [12]
LIP_DOWN = [14]

# handle of the webcam
cap = cv.VideoCapture(0)

arr = []

#print("Centering Mouse In Screen...")
X = int(get_monitors()[0].width / 2)
Y = int(get_monitors()[0].height / 2)

# center mouse on program init...
mouse.position = (X, Y)

Y_BUFFER = []
LEFT_EYE_HEIGHT_BUFF = []
RIGHT_EYE_HEIGHT_BUFF = []
LEFT_EYE_HEIGHT_BUFF_DYN = []
RIGHT_EYE_HEIGHT_BUFF_DYN = []

mouseEnabled = True 
actionPerformed = False
previous = time()
delta = 0

# Mediapipe parametes
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while mouseEnabled:
        
        # measure elapsed time
        current = time()
        delta += current - previous
        previous = current
        
        # enable mouse click again
        if (delta > 2):
            actionPerformed = False
        
        key = cv.waitKey(10)
        if key == ord('q'):
            sys.exit("Program Terminated")
            print('Q Pressed')
            break

        # read current frame            
        ret, frame = cap.read()
        if not ret:
            break

        # process frame with mediapipe
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        # if IRIS landmarks found...
        if results.multi_face_landmarks:

            # collect all mesh points
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
            for p in results.multi_face_landmarks[0].landmark])

            # esimate IRIS circle+
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            # get iris center coordinates
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            cv.circle(frame, center_left, int(2), (255,255,0), -1, cv.FILLED)
            cv.circle(frame, center_right, int(2), (255,255,0), -1, cv.FILLED)
            
            # draw nose triangle
            nose = np.array([mesh_points[NOSE_CENTER][0][0], mesh_points[NOSE_CENTER][0][1]], dtype=np.int32) 
            cv.circle(frame, nose, int(2), (255,255,0), -1, cv.FILLED)
        
            cv.line(frame, center_left, nose, (255, 0, 0), 1)
            cv.line(frame, nose, center_right, (255, 0, 0), 1)
            cv.line(frame, center_right, center_left, (255, 0, 0), 1)
           
            # mouse movement
            X, Y = move_mouse_x(nose, center_right, center_left, X, Y, mouse)
            X, Y = move_mouse_y(nose, center_right, center_left, X, Y, mouse, Y_BUFFER)
            
            # eyes closing 
            right_eye = mesh_points[RIGHT_EYE]
            left_eye = mesh_points[LEFT_EYE]
            
            # estimate eye-height for each eye
            len_left = open_len(right_eye)
            len_right = open_len(left_eye)
            
            # create buffer of eye socket size...
            mean_eye_left, LEFT_EYE_HEIGHT_BUFF = buffer_set(LEFT_EYE_HEIGHT_BUFF, len_left, 30, False)
            mean_eye_right, RIGHT_EYE_HEIGHT_BUFF = buffer_set(RIGHT_EYE_HEIGHT_BUFF, len_right, 30, False)
            
            # create a dynamic buffer of current eye socket size....
            mean_eye_left_dyn, LEFT_EYE_HEIGHT_BUFF_DYN = buffer_set(LEFT_EYE_HEIGHT_BUFF_DYN, len_left, 30)
            mean_eye_right_dyn, RIGHT_EYE_HEIGHT_BUFF_DYN = buffer_set(RIGHT_EYE_HEIGHT_BUFF_DYN, len_right, 30)
            
            # determine if eyes are closed or opened...
            left_closed = eye_closed(mean_eye_left_dyn, LEFT_EYE_HEIGHT_BUFF, 30)
            right_closed = eye_closed(mean_eye_right_dyn, RIGHT_EYE_HEIGHT_BUFF, 30)
            
            
            # mouth open is for double click
            lip_up = mesh_points[LIP_UP]
            lip_down = mesh_points[LIP_DOWN]
            
            mouth_opened = mouth_open(lip_up[0][1], lip_down[0][1], 80)
            

            if (mouth_opened and actionPerformed == False and delta > 2):
                print("double click")
                LEFT_EYE_HEIGHT_BUFF_DYN = []
                RIGHT_EYE_HEIGHT_BUFF_DYN = []
                mouse.click(Button.left, 2)
                actionPerformed = True
                delta = 0
                right_closed = False
                left_closed = False
                mouth_opened = False
                continue
            
            if (right_closed and actionPerformed == False and delta > 2):
                print("right click")
                LEFT_EYE_HEIGHT_BUFF_DYN = []
                RIGHT_EYE_HEIGHT_BUFF_DYN = []
                mouse.press(Button.right)
                mouse.release(Button.right)
                actionPerformed = True
                delta = 0
                right_closed = False
                left_closed = False
                mouth_opened = False
                continue
               
                
            if (left_closed and actionPerformed == False and delta > 2):
                print("left click")
                LEFT_EYE_HEIGHT_BUFF_DYN = []
                RIGHT_EYE_HEIGHT_BUFF_DYN = []
                mouse.press(Button.left)
                mouse.release(Button.left)
                actionPerformed = True
                delta = 0
                right_closed = False
                left_closed = False
                mouth_opened = False
                continue

        #cv.imshow('Mediapipe Eye Gaze Tracking', frame)

cap.release()
cv.destroyAllWindows()