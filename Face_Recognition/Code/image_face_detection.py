# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:40:28 2024

@author: Vivek Mishra
"""

#import required packages
import cv2
import tensorflow as tf
import face_recognition
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json,load_model
import numpy as np
import face_recognition

#Load the image to detect
image_to_detect=cv2.imread(r'C:/Users/vivek/OneDrive/Documents/Face_Recognition/images/testing/trump-modi.jpg')
#cv2.imshow("Test", image_to_detect)
#face expression model initialization

face_exp_model=load_model(r'C:/Users/vivek/OneDrive/Documents/Face_Recognition/Dataset/facial_expression_model_combined.h5')

#List of emotions Labels
emotions_label=('Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral')

#detect all face in the image
all_face_locations = face_recognition.face_locations(image_to_detect,model='hog')
#printing the number of faces in the array
print('There are {} faces in this image'.format(len(all_face_locations)))

#looping through the face loactions

for index,current_face_location in enumerate(all_face_locations):
  #splitting the tuple to get the four position values of current face
  top_pos,right_pos,bottom_pos,left_pos = current_face_location
  #Printing the location of current face
  print('Found face {} at top:{},right:{},bottom:{},left:{}'.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
  #Slicing image by position inside the loop
  current_face_image = image_to_detect[top_pos:bottom_pos,left_pos:right_pos]
  #Draw rectangle around the face detected
  cv2.rectangle(image_to_detect,(left_pos,top_pos),(right_pos,bottom_pos),(1,255,25), 2)
  
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
  text_position = (left_pos, top_pos - 11)  # Position the text above the rectangle
  text_color = (0, 0, 255)  # Red color (BGR format)
  font_scale = 1.2  # Slightly larger text size for bold effect
  font_thickness = 1  # Bold effect
  cv2.putText(image_to_detect, emotion_label, text_position, font, font_scale, text_color, font_thickness)


#showing the current face with the rectangle drawn
cv2.imshow("Image Face Emotions",image_to_detect)

cv2.waitKey(0)  # Wait indefinitely for a key press
cv2.destroyAllWindows()  # Close all windows after the key press


