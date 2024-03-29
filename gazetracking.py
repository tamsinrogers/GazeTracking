"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import csv
from gaze_tracking import GazeTracking
import time
from sklearn.linear_model import LinearRegression
import numpy as np
import math
import mediapipe as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import os

import keras
import tensorflow
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from keras.applications.mobilenet import MobileNet, preprocess_input 
from keras.losses import categorical_crossentropy
from keras.models import load_model
import joblib

from PIL import Image
import glob

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)


launch_time = time.time()
duration = 3


# We get a new frame from the webcam
_, frame = webcam.read()

pupil1 = (0, 0)
pupil2 = (0, 0)
pupil3 = (0, 0)
pupil4 = (0, 0)
pupil5 = (0, 0)
pupil6 = (0, 0)
pupil7 = (0, 0)
pupil8 = (0, 0)
pupil9 = (0, 0)
pupil10 = (0, 0)

point1 = (0,0)
point2 = (0, 0)
point3 = (0, 0)
point4 = (0, 0)
point5 = (0, 0)
point6 = (0, 0)
point7 = (0, 0)
point8 = (0, 0)
point9 = (0, 0)
point10 = (0, 0)

not_calibrated = True
not_validated = True

'''MAIN FUNCTION'''
def main():

	window_x, window_y, window_width, window_height = 0, 0, 1280, 800

	not_calibrated = True
	not_validated = True

	current_frame_count = 0
	
	buffer = []	# every time we add to this, we remove one (always n=10)


	while True:

		with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:

			# We get a new frame from the webcam
			_, frame = webcam.read()

			# We send this frame to GazeTracking to analyze it
			gaze.refresh(frame)

			frame = gaze.annotated_frame()
		
			results = face_mesh.process(frame)

			frame.flags.writeable = True
		
			if not os.path.exists('frames'):
				os.mkdir('frames')
		
			if results.multi_face_landmarks:
				for face_landmarks in results.multi_face_landmarks:
					mp_drawing.draw_landmarks(image=frame, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=drawing_spec, connection_drawing_spec= drawing_spec)
				
				
			# STORE FRAMES
			name = 'frames/' + str(current_frame_count) + '.jpg'
			cv2.imwrite(name, frame)
			current_frame_count += 1

			dim = (window_width, window_height)
			frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
			frame = cv2.flip(frame, 1)
			cv2.imshow("Demo", frame)
		

			left_pupil = gaze.pupil_left_coords()
			right_pupil = gaze.pupil_right_coords()
			
			buffer.append(right_pupil)	# train the model with ten samples

			# Radius of circle
			fixation_radius = 10
			calibration_radius = 30

			# color in BGR
			fixation_color = (0, 255, 0)
			calibration_color = (255, 0, 0)

			# Line thickness
			fixation_thickness = 3
			calibration_thickness = 15

			# seconds
			duration = 3 

			calibration()
			calculateCalibrationModel(True)
			gazeEstimation()
			validation(launch_time= time.time(), duration=3)
			dataCalculations(True)
	
		cv2.namedWindow("Demo", cv2.WND_PROP_FULLSCREEN)	
		cv2.setWindowProperty("Demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		cv2.imshow("Demo", frame)
		window_x, window_y, window_width, window_height = cv2.getWindowImageRect('Demo')
		cv2.imshow("Demo", frame)

		# escape pressed to quit
		if cv2.waitKey(1) == 27:
			break

		# C pressed to recalibrate
		if cv2.waitKey(1) == ord('c'):
			print("C pressed: recalibrating!")
			not_calibrated = True
			launch_time = time.time()
			calibration()
		
		# V pressed to recalibrate
		if cv2.waitKey(1) == ord('v'):
			print("V pressed: revalidating!")
			not_validated = True
			launch_time = time.time()
			validation()

		webcam.release()
		cv2.destroyAllWindows()
	
	
def reset():
	pupil1 = (0, 0)
	pupil2 = (0, 0)
	pupil3 = (0, 0)
	pupil4 = (0, 0)
	pupil5 = (0, 0)
	pupil6 = (0, 0)
	pupil7 = (0, 0)
	pupil8 = (0, 0)
	pupil9 = (0, 0)
	pupil10 = (0, 0)
	
	point1 = (0,0)
	point2 = (0, 0)
	point3 = (0, 0)
	point4 = (0, 0)
	point5 = (0, 0)
	point6 = (0, 0)
	point7 = (0, 0)
	point8 = (0, 0)
	point9 = (0, 0)
	point10 = (0, 0)

	not_calibrated = True
	not_validated = True

	current_frame_count = 0


'''RUNS MODEL FROM PICKLED MODEL OR TRAINING FROM SCRATCH'''
def runModel():
	if not os.path.exists("best_model.h5"):		# first time running, no pickled model
		print("TRAINING FROM SCRATCH")
		train()

	else:										# using pickled model
		print("USING PICKLED MODEL")


'''STORES USER'S NAME'''
def inputName():
	n = input()
	return n

'''TRAINS NEURAL NETWORK FROM SCRATCH'''
def train():

	# a type of convolutional neural network designed for mobile and embedded vision applications
	base_model = MobileNet( input_shape=(224,224,3), include_top=False )	
	
	for layer in base_model.layers:
	  layer.trainable = False

	x = Flatten()(base_model.output)
	x = Dense(units=7 , activation='softmax' )(x)

	# creating  model
	model = Model(base_model.input, x)
	model.compile(optimizer='adam', loss= categorical_crossentropy , metrics=['accuracy']  )

	# prepare data
	train_datagen = ImageDataGenerator(zoom_range = 0.2, shear_range = 0.2, horizontal_flip=True, rescale = 1./255)
	train_data = train_datagen.flow_from_directory(directory= "train", target_size=(224,224), batch_size=32,)
	train_data.class_indices

	val_datagen = ImageDataGenerator(rescale = 1./255 )
	val_data = val_datagen.flow_from_directory(directory= "train", target_size=(224,224), batch_size=32,)

	# early stopping & model checkpoint
	from keras.callbacks import ModelCheckpoint, EarlyStopping

	# early stopping
	es = EarlyStopping(monitor='val_accuracy', min_delta= 0.01 , patience= 5, verbose= 1, mode='auto')

	# model check point
	mc = ModelCheckpoint(filepath="best_model.h5", monitor= 'val_accuracy', verbose= 1, save_best_only= True, mode = 'auto')

	# puting call back in a list 
	call_back = [es, mc]

	hist = model.fit_generator(train_data, 
							   steps_per_epoch= 10, 
							   epochs= 30, 
							   validation_data= val_data, 
							   validation_steps= 8, 
							   callbacks=[es,mc])

'''PREDICTS EMOTION FROM IMAGE IN PATH'''
def emotion(path, loaded_model):

	img = load_img(path, target_size=(224,224) )	# path for the image to see if it predics correct class

	i = img_to_array(img)/255
	input_arr = np.array([i])
	input_arr.shape
	
	# prepare data
	train_datagen = ImageDataGenerator(zoom_range = 0.2, shear_range = 0.2, horizontal_flip=True, rescale = 1./255)
	train_data = train_datagen.flow_from_directory(directory= "train", target_size=(224,224), batch_size=32,)
	train_data.class_indices
	
	op = dict(zip( train_data.class_indices.values(), train_data.class_indices.keys()))
	
	pred = np.argmax(loaded_model.predict(input_arr))
	return op[pred]


'''MANAGES TIME BETWEEN PUPIL POSITION RECORDINGS'''
def within_time(launch_time, start, end):

	current_time = time.time()

	if launch_time + start < current_time < launch_time + end:
		return True

	return False
	

'''CALIBRATION PHASE'''
def calibration():

	not_calibrated = True
	not_validated = True

	pupil1 = (0, 0)
	pupil2 = (0, 0)
	pupil3 = (0, 0)
	pupil4 = (0, 0)
	pupil5 = (0, 0)
	pupil6 = (0, 0)
	pupil7 = (0, 0)
	pupil8 = (0, 0)
	pupil9 = (0, 0)
	pupil10 = (0, 0)

	point1 = (0,0)
	point2 = (0, 0)
	point3 = (0, 0)
	point4 = (0, 0)
	point5 = (0, 0)
	point6 = (0, 0)
	point7 = (0, 0)
	point8 = (0, 0)
	point9 = (0, 0)
	point10 = (0, 0)

	cv2.putText(frame, "ENTERING CALIBRATION PHASE", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 1)
	launch_time = time.time()
	
	window_x, window_y, window_width, window_height = 0, 0, 1280, 800

	# top left
	point1 = (50, 50)
	if within_time(launch_time, duration, duration*2):
		cv2.circle(frame, point1, calibration_radius, calibration_color, calibration_thickness)
		pupil1 = gaze.pupil_right_coords()	 # becomes an array of 10 frames of xy coordinates
		# pupil1 = buffer

	# top right
	point2 = ( window_width - 50, 50)
	if within_time(launch_time, duration*2, duration*3):
		cv2.circle(frame, point2, calibration_radius, calibration_color, calibration_thickness)
		pupil2 = gaze.pupil_right_coords()

	# bottom right
	point3 = ( window_width - 50, window_height - 50)
	if within_time(launch_time, duration*3, duration*4):
		cv2.circle(frame, point3, calibration_radius, calibration_color, calibration_thickness)
		pupil3 = gaze.pupil_right_coords()

	# bottom left
	point4 = ( 50, window_height - 50)
	if within_time(launch_time, duration*4, duration*5):
		cv2.circle(frame, point4, calibration_radius, calibration_color, calibration_thickness)
		pupil4 = gaze.pupil_right_coords()

	# center
	point5 = ( window_width//2, window_height//2)
	if within_time(launch_time, duration*5, duration*6):
		cv2.circle(frame, point5, calibration_radius, calibration_color, calibration_thickness)
		pupil5 = gaze.pupil_right_coords()

	point6 = (window_width//2, window_height//4)
	if within_time(launch_time, duration*6, duration*7):
		cv2.circle(frame, point6, calibration_radius, calibration_color, calibration_thickness)
		pupil6 = gaze.pupil_right_coords()

	point7 = ( window_width//2, (window_height//4)*3)
	if within_time(launch_time, duration*7, duration*8):
		cv2.circle(frame, point7, calibration_radius, calibration_color, calibration_thickness)
		pupil7 = gaze.pupil_right_coords()

	point8 = ( window_width//4, (window_height//4)*3)
	if within_time(launch_time, duration*8, duration*9):
		cv2.circle(frame, point8, calibration_radius, calibration_color, calibration_thickness)
		pupil8 = gaze.pupil_right_coords()

	point9 = ( (window_width//4)*3, window_height//4)
	if within_time(launch_time, duration*9, duration*10):
		cv2.circle(frame, point9, calibration_radius, calibration_color, calibration_thickness)
		pupil9 = gaze.pupil_right_coords()

	point10 = ( (window_width//4)*3, (window_height//4)*3)
	if within_time(launch_time, duration*10, duration*11):
		cv2.circle(frame, point10, calibration_radius, calibration_color, calibration_thickness)
		pupil10 = gaze.pupil_right_coords()

	calibrationPoints = [pupil1, pupil2, pupil3, pupil4, pupil5, pupil6, pupil7, pupil8, pupil9, pupil10]

	if None in calibrationPoints:
		# recalibrate
		print("None in calibration points, recalibrating!")
		not_calibrated = True
		launch_time = time.time()

'''CALCULATES MODEL TO BE USED FOR GAZE ESTIMATION'''
def calculateCalibrationModel(not_calibrated):

	reg_x = 0
	reg_y = 0

	if not_calibrated and within_time(launch_time, duration*11, duration*12):
		not_calibrated = False

		x = np.array([list(pupil1), list(pupil2), list(pupil3), list(pupil4), list(pupil5), list(pupil6), list(pupil7), list(pupil8), list(pupil9), list(pupil10)])
		x_targets = np.array([point1[0], point2[0], point3[0], point4[0],point5[0], point6[0], point7[0], point8[0], point9[0], point10[0]])

		y = np.array([list(pupil1), list(pupil2), list(pupil3), list(pupil4), list(pupil5), list(pupil6), list(pupil7), list(pupil8), list(pupil9), list(pupil10)])
		y_targets = np.array([point1[1], point2[1], point3[1], point4[1],point5[1], point6[1], point7[1], point8[1], point9[1], point10[1]])

		reg_x = LinearRegression().fit(x, x_targets)
		reg_y = LinearRegression().fit(y, y_targets)

	return reg_x, reg_y

'''CALCULATES XY VALUES TO BE USED FOR GAZE ESTIMATION'''
def gazeEstimation():

	reg_x, reg_y = calculateCalibrationModel(True)

	# gaze estimation after calibration
	if not not_calibrated:

		if gaze.pupil_right_coords() is not None:
			# predict display x, y

			display_x = reg_x.predict(np.array([list(gaze.pupil_right_coords())]))	# getBuffer() instead
			display_y = reg_y.predict(np.array([list(gaze.pupil_right_coords())]))

		# gaze point
		fixation = (int(display_x), int(display_y))

		# draw x, y on screen
		cv2.circle(frame, fixation, fixation_radius, fixation_color, fixation_thickness)

'''VALIDATION PHASE'''
def validation(launch_time, duration):

	window_x, window_y, window_width, window_height = 0, 0, 1280, 800

	cv2.putText(frame, "CALIBRATION COMPLETE", (90, 200), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 1)
	cv2.putText(frame, "ENTERING VALIDATION PHASE", (90, 270), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 1)

	# draw the circles

	# top left
	point1 = (50, 50)
	if within_time(launch_time, duration*15, duration*16):
		cv2.circle(frame, point1, calibration_radius, calibration_color, calibration_thickness)
		pupil1 = gaze.pupil_right_coords()

	# top right
	point2 = ( window_width - 50, 50)
	if within_time(launch_time, duration*16, duration*17):
		cv2.circle(frame, point2, calibration_radius, calibration_color, calibration_thickness)
		pupil2 = gaze.pupil_right_coords()

	# bottom right
	point3 = ( window_width - 50, window_height - 50)
	if within_time(launch_time, duration*17, duration*18):
		cv2.circle(frame, point3, calibration_radius, calibration_color, calibration_thickness)
		pupil3 = gaze.pupil_right_coords()

	# bottom left
	point4 = ( 50, window_height - 50)
	if within_time(launch_time, duration*18, duration*19):
		cv2.circle(frame, point4, calibration_radius, calibration_color, calibration_thickness)
		pupil4 = gaze.pupil_right_coords()

	# center
	point5 = ( window_width//2, window_height//2)
	if within_time(launch_time, duration*19, duration*20):
		cv2.circle(frame, point5, calibration_radius, calibration_color, calibration_thickness)
		pupil5 = gaze.pupil_right_coords()

	point6 = (window_width//2, window_height//4)
	if within_time(launch_time, duration*20, duration*21):
		cv2.circle(frame, point6, calibration_radius, calibration_color, calibration_thickness)
		pupil6 = gaze.pupil_right_coords()

	point7 = ( window_width//2, (window_height//4)*3)
	if within_time(launch_time, duration*21, duration*22):
		cv2.circle(frame, point7, calibration_radius, calibration_color, calibration_thickness)
		pupil7 = gaze.pupil_right_coords()

	point8 = ( window_width//4, (window_height//4)*3)
	if within_time(launch_time, duration*22, duration*23):
		cv2.circle(frame, point8, calibration_radius, calibration_color, calibration_thickness)
		pupil8 = gaze.pupil_right_coords()

	point9 = ( (window_width//4)*3, window_height//4)
	if within_time(launch_time, duration*23, duration*24):
		cv2.circle(frame, point9, calibration_radius, calibration_color, calibration_thickness)
		pupil9 = gaze.pupil_right_coords()

	point10 = ( (window_width//4)*3, (window_height//4)*3)
	if within_time(launch_time, duration*24, duration*25):
		cv2.circle(frame, point10, calibration_radius, calibration_color, calibration_thickness)
		pupil10 = gaze.pupil_right_coords()

		validationPoints = [pupil1, pupil2, pupil3, pupil4, pupil5, pupil6, pupil7, pupil8, pupil9, pupil10]
		actualPoints = [point1, point2, point3, point4, point5, point6, point7, point8, point9, point10]

		if None in validationPoints:
			# revalidate
			print("None in validation points, recalibrating!")
			not_validated = True
			launch_time = time.time()
			

'''CALCULATE ACCURACY AND EMOTION, WRITE TO CSV'''
def dataCalculations(not_validated):

	# calculate euclidean distance between points
	if not_validated and within_time(launch_time, duration*26, duration*40):
		not_validated = False

		cv2.putText(frame, "VALIDATION COMPLETE", (90, 340), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 1)
		cv2.putText(frame, "CALCULATING ACCURACY", (90, 410), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 1)

		experimental = np.array([ list(pupil1), list(pupil2), list(pupil3), list(pupil4), list(pupil5), list(pupil6), list(pupil7), list(pupil8), list(pupil9), list(pupil10) ])
		actual = np.array([ list(point1), list(point2), list(point3), list(point4), list(point5), list(point6), list(point7), list(point8), list(point9), list(point10) ])

		distances = []
		
		f = inputName() + ".csv"

		with open(f, 'w', newline='') as file:
		
			csvreader = csv.reader(file)
			writer = csv.writer(file)
			writer.writerow(["Actual Coordinate", "Experimental Coordinate", "Accuracy", "Emotion Detected"])
	
			# write data to csv
			for i in range(len(actual)):
																			# write set of points to text file
				d = math.dist(actual[i], experimental[i])					# calculate dist for each set of points
				distances.append(d)											# add to total distances
				path = 'frames/' + str(i) + '.jpg' 								
				e = emotion(path, loaded_model)								# determine emotion at each frame
				writer.writerow([actual[i], experimental[i], d, str(e)]) 	# write to csv 
									
			# average euclidean distance calculation
			avgDist = sum(distances)/(len(distances))
			writer.writerow(["AVG", "AVG", avgDist])
		
		cv2.putText(frame, "CALCULATING ACCURACY", (90, 480), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,255,255), 1)
									

if __name__ == "__main__":
    main()