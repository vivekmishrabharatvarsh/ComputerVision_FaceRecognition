# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 22:56:54 2024

@author: Vivek Mishra
"""

#import required packages
import cv2
import tensorflow as tf
import face_recognition
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json,load_model
import numpy as np






webcam_video_stream= cv2.VideoCapture(r'C:/Users/vivek/OneDrive/Documents/Face_Recognition/images/samples/videoplayback.webm')

#face expression model initialization

face_exp_model=load_model(r'C:/Users/vivek/OneDrive/Documents/Face_Recognition/Dataset/facial_expression_model_combined.h5')

#List of emotions Labels
emotions_label=('Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral')
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

        #The AGE_GENDER_MEAN_VALUES calculated by using the numpy.mean()
        AGE_GENDER_MEAN_VALUES=(78.4264,87.7689,114.8958)

        #Create blob of current face slice
        current_face_image_blob= cv2.dnn.blobFromImage(current_face_image,1,(227,227),AGE_GENDER_MEAN_VALUES,swapRB=False)
        
        #Gender Parameters
        gender_label_list=['Male','Female']
        gender_protext=r"C:/Users/vivek/OneDrive/Documents/Face_Recognition/Dataset/gender_deploy.prototxt"
        gender_caffemodel=r"C:/Users/vivek/OneDrive/Documents/Face_Recognition/Dataset/gender_net.caffemodel"
        gender_cov_net= cv2.dnn.readNet(gender_caffemodel, gender_protext)
        gender_cov_net.setInput(current_face_image_blob)
        gender_predictions=gender_cov_net.forward()
        gender=gender_label_list[gender_predictions[0].argmax()]

        #Age Parameters
        age_label_list = ['(0-5)', '(5-10)', '(10-15)', '(15-20)', '(20-25)', '(25-30)', '(30-35)', '(35-40)', '(40-45)', '(45-50)', '(50-55)', '(55-60)', '(60-65)', '(65-70)', '(70-75)', '(75-80)', '(80-85)', '(85-90)', '(90-95)', '(95-100)']
        age_protext=r"C:/Users/vivek/OneDrive/Documents/Face_Recognition/Dataset/age_deploy.prototxt"
        age_caffemodel=r"C:/Users/vivek/OneDrive/Documents/Face_Recognition/Dataset/age_net.caffemodel"
        age_cov_net= cv2.dnn.readNet(age_caffemodel, age_protext)
        age_cov_net.setInput(current_face_image_blob)
        age_predictions=age_cov_net.forward()
        age=age_label_list[age_predictions[0].argmax()]

        #Draw rectangle around the face detected
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(1,255,25), 2)
        
        #Preprocess input, convert it to an image like as the data in dataset
        #Convert to GrayScale
        current_face_image=cv2.cvtColor(current_face_image,cv2.COLOR_BGR2GRAY)

        #resize to 48X48 pixel size
        current_face_image=cv2.resize(current_face_image,(48,48))

        #convert the PIL image into a 3d numpy array
        img_pixels=image.img_to_array(current_face_image)

        #Expand the shape of an array into single row multiple columns
        img_pixels=np.expand_dims(img_pixels,axis=0)

        #pixels are in range of [0,255], normalize all pixels in scale of [0,1]
        #img_pixels/=255

        #do prediction using model, get the prediction values for all 7 expressions
        exp_predictions=face_exp_model.predict(img_pixels)

        #find max indexed predictions value (0 till 7)
        max_index= np.argmax(exp_predictions[0])

        #get corresponding lablel from emotional_label
        emotion_label= emotions_label[max_index]

        #display the name as text in the image
        # Display the name as text outside the rectangle box in red and bold
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_position = (left_pos, top_pos - 80)  # Position the text above the rectangle
        text_color = (0, 0, 255)  # Red color (BGR format)
        font_scale = 1.2  # Slightly larger text size for bold effect
        font_thickness = 2  # Bold effect
        # Define text details
        text_line1 = f"{gender} {age} years"
        text_line2 = emotion_label

        # Define the vertical spacing between lines
        line_spacing = 50  

        # Draw the first line
        cv2.putText(current_frame, text_line1, text_position, font, font_scale, text_color, font_thickness)

        # Draw the second line slightly below the first
        text_position_line2 = (text_position[0], text_position[1] + line_spacing)
        cv2.putText(current_frame, text_line2, text_position_line2, font, font_scale, text_color, font_thickness)

        #cv2.putText(current_frame,gender+" "+age+" " +"years"+'\n'+emotion_label, text_position, font, font_scale, text_color, font_thickness)

    # Resize the frame to a fixed resolution (640x480)
    desired_width = 490
    desired_height = 690
    current_frame = cv2.resize(current_frame, (desired_width, desired_height))
    #showing the current face with the rectangle drawn
    cv2.imshow("Output Video",current_frame)
   
    if cv2.waitKey(1) & 0xFF==ord('q'):
        
        break

webcam_video_stream.release()
cv2.destroyAllWindows()








