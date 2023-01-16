"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import time
from sklearn.linear_model import LinearRegression
import numpy as np

def within_time(launch_time, start, end):

    current_time = time.time()

    if launch_time + start < current_time < launch_time + end:
        return True

    return False


gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

launch_time = time.time()
window_x, window_y, window_width, window_height = 0, 0, 1280, 800

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

not_calibrated = True

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    dim = (window_width, window_height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Demo", frame)

    # if gaze.is_blinking():
    #     text = "Blinking"
    # elif gaze.is_right():
    #     text = "Looking right"
    # elif gaze.is_left():
    #     text = "Looking left"
    # elif gaze.is_center():
    #     text = "Looking center"

    #cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    #cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    #cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

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
    
    ''' CALIBRATION POINTS '''
    
    #pupil1 = get_pupil_in_time(launch_time, 2, 4, gaze.pupil_right_coords())

    # top left
    point1 = (50, 50)
    if within_time(launch_time, duration, duration*2):
        cv2.circle(frame, point1, calibration_radius, calibration_color, calibration_thickness)
        pupil1 = gaze.pupil_right_coords()

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

    x = [pupil1, pupil2, pupil3, pupil4, pupil5, pupil6, pupil7, pupil8, pupil9, pupil10]

    if None in x:
        # recalibrate
        print("None in calibration points, recalibrating!")
        not_calibrated = True
        launch_time = time.time()


    # calculate calibration model
    if not_calibrated and within_time(launch_time, duration*11, duration*12):
        not_calibrated = False
        
        x = np.array([list(pupil1), list(pupil2), list(pupil3), list(pupil4), list(pupil5), list(pupil6), list(pupil7), list(pupil8), list(pupil9), list(pupil10)])
        x_targets = np.array([point1[1], point2[1], point3[1], point4[1],point5[1], point6[1], point7[1], point8[1], point9[1], point10[1]])

        y = np.array([list(pupil1), list(pupil2), list(pupil3), list(pupil4), list(pupil5), list(pupil6), list(pupil7), list(pupil8), list(pupil9), list(pupil10)])
        y_targets = np.array([point1[1], point2[1], point3[1], point4[1],point5[1], point6[1], point7[1], point8[1], point9[1], point10[1]])

        # run regression 
        reg_x = LinearRegression().fit(x, x_targets)
        reg_y = LinearRegression().fit(y, y_targets)

    # gaze estimation after calibration
    if not not_calibrated:

        if gaze.pupil_right_coords() is not None:
            # predict display x, y
            display_x = reg_x.predict(np.array([list(gaze.pupil_right_coords())]))
            display_y = reg_y.predict(np.array([list(gaze.pupil_right_coords())]))

        # gaze point
        fixation = (int(display_x), int(display_y))

        # draw x, y on screen
        # Using cv2.circle() method
        cv2.circle(frame, fixation, fixation_radius, fixation_color, fixation_thickness)


    cv2.namedWindow("Demo", cv2.WND_PROP_FULLSCREEN)    
    cv2.setWindowProperty("Demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    frame = cv2.flip(frame, 1)
    cv2.imshow("Demo", frame)

    window_x, window_y, window_width, window_height = cv2.getWindowImageRect('Demo')
    cv2.imshow("Demo", frame)

    #print(cv2.getWindowImageRect('Demo'))

    # escape pressed to quit
    if cv2.waitKey(1) == 27:
        break

    # c pressed to recalibrate
    if cv2.waitKey(1) == 99:
        print("C pressed: recalibrating!")
        not_calibrated = True
        launch_time = time.time()
   
webcam.release()
cv2.destroyAllWindows()
