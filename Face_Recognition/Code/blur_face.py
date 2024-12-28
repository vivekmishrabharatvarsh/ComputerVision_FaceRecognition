# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:56:54 2024

@author: Vivek Mishra
"""

#import required packages
import cv2
import face_recognition

webcam_video_stream= cv2.VideoCapture(r'C:/Users/vivek/OneDrive/Documents/Face_Recognition/images/testing/modi.mp4')

#initialize the array variable to hold all face location in the frame
all_face_locations=[]

#loop through every frame in the video

while True:
    #get the current frame from the video stream as an image
    ret,current_frame=webcam_video_stream.read()
    #resize the current frame to 1/4 size to process faster
    current_frame_small=cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #detect all faces in the image
    #arguments are image,no_of_times_to_unsample,model
    all_face_locations=face_recognition.face_locations(current_frame_small,model='hog')
    for index,current_face_location in enumerate(all_face_locations):
        #splitting the tuple to get the four position values of current face
        top_pos,right_pos,bottom_pos,left_pos = current_face_location
        
        #changing the position magnitude to fit the actual size video frame
        top_pos*=4
        right_pos*=4
        bottom_pos*=4
        left_pos*=4
        
        #Printing the location of current face
        print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        
        #Slicing image by position inside the loop
        current_face_image = current_frame[top_pos:bottom_pos,left_pos:right_pos]
        #Blur the sliced face and save it to the same array itself
        current_face_image=cv2.GaussianBlur(current_face_image,(99,99),30)
        
        #Paste the blurred face into the actual frame
        current_frame[top_pos:bottom_pos,left_pos:right_pos]=current_face_image

        #Draw rectangle around the face detected
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(1,255,25), 2)
        
    #showing the current face with the rectangle drawn
    cv2.imshow("Webcam Video",current_frame)
   
    if cv2.waitKey(1) & 0xFF==ord('q'):
        
        break

webcam_video_stream.release()
cv2.destroyAllWindows()








