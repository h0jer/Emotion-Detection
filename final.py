# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:39:42 2024

@author: fagr
"""

from deepface import DeepFace
import cv2
import numpy as np
import dlib
from math import hypot
import time
import streamlit as st

font = cv2.FONT_HERSHEY_PLAIN


def midpoint(p1 ,p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_eye_ration(points , landmarks):
        # draw hirozontal line
        left_point =(landmarks.part(points[0]).x , landmarks.part(points[0]).y)
        right_point =(landmarks.part(points[3]).x , landmarks.part(points[3]).y)
        cv2.line(frame , left_point , right_point , (0,255,0),2)
        
        # draw vertical line 
        top_center = midpoint(landmarks.part(points[1]) , landmarks.part(points[2]))
        bottom_center = midpoint(landmarks.part(points[5]) , landmarks.part(points[4]))
        cv2.line(frame , top_center , bottom_center , (0,255,0) ,2)
        
        ver_line_lenght = hypot((top_center[0] - bottom_center[0]) , (top_center[1] - bottom_center[1]))
        her_line_lenght = hypot((left_point[0] - right_point[0]) ,(left_point[1] - right_point[1]) )
        ratio = her_line_lenght / ver_line_lenght
        return ratio
    


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\fagr\Downloads\shape_predictor_68_face_landmarks.dat")

st.title("Real-Time Detection")
run = st.checkbox('Run')
frame_placeholder = st.empty()


cap = cv2.VideoCapture(0)
eye_closed_start = None
while run:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        cv2.putText(frame, "No student", (50, 150), font, 2, (0, 0, 255))
        
    for face in faces:
        predictions = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        text = predictions[0]['dominant_emotion']
        cv2.putText(frame, text, (50, 100), font, 2, (0, 0, 255))
        
        # determine the rectangular points and draw it (face detection)
        x1 , y1 = face.left() , face.top()
        x2 , y2 = face.right() , face.bottom()
        cv2.rectangle(frame, (x1,y1) , (x2,y2), (0,255,0) ,2)
        
        
        # determine sleep or not
        landmarker = predictor(gray , face)
        left_eye_ratio = get_eye_ration([36,37 , 38 , 39 , 40 ,41] , landmarker)
        right_eye_ratio = get_eye_ration([42 , 43 , 44 , 45 , 46 ,47] , landmarker)
        
        ratio = (left_eye_ratio + right_eye_ratio)/2
         
        if ratio > 4:    
            if eye_closed_start is None:
                eye_closed_start = time.time()  # Start the timer
            elif time.time() - eye_closed_start >= 5:
                cv2.putText(frame, "Sleeping", (50, 150), font, 2, (0, 0, 255))  # Alert for sleep
        else:
            eye_closed_start = None
    
    
    
    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    
cap.release()
 