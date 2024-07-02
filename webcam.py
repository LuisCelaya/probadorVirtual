import cv2
import time
import mediapipe as mp
import numpy as np
from utils import *

x = [0,0,0,0,0]
y = [0,0,0,0,0]

i = 0

# interes = [162,389,234,454,6,35,265,8]
interes = [162,389,234,454,6]

imagen = cv2.imread("./src/gafas3.png", cv2.IMREAD_UNCHANGED)
dts = imagen

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
segunda = cap
time.sleep(2)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:

    while True:
        ret,frame = cap.read()
        _ ,segundo = segunda.read()

        height, width, _ = frame.shape
        frame = cv2.flip(frame,1)
        segundo = cv2.flip(segundo,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results  = face_mesh.process(frame_rgb)
        
        interes = [71,301,116,345,5]

        
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                for inter in interes:    
                    
                    x[i] = int(face_landmarks.landmark[inter].x * width) 
                    y[i] = int(face_landmarks.landmark[inter].y * height)   
                    
                    i = i + 1
                i = 0    

            pts_src = np.array([
                [0, 0],
                [512, 0],
                [0, 160],
                [512, 160],
                [256, 90]
                ], dtype=np.float32)
            
            hola = [x[0], y[0]]
            que = [x[1], y[1]]
            tal = [x[2], y[2]]
            prueba = [x[3], y[3]]
            vital = [x[4], y[4]]

            pts_dst = np.array([hola, que, tal, prueba, vital], dtype=np.float32)

            H, _ = cv2.findHomography(pts_src, pts_dst)
            dts = cv2.warpPerspective(imagen, H, (1000,1000))
            otra = Util.overlay(frame, dts)

        cv2.imshow('imagen', dts)
        cv2.imshow('WebCam', frame)

        if cv2.waitKey(1)&0xFF == ord('q') or cv2.getWindowProperty('WebCam', cv2.WND_PROP_VISIBLE) < 1:
            break


cap.release()
cv2.destroyAllWindows()     

