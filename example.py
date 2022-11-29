"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking

import pygame

pygame.init()

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

frameWidth = int((webcam.get(cv2.CAP_PROP_FRAME_WIDTH)))
frameHeight = int((webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)))





while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    
    
    
    text = ""

    '''
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"
    '''

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    
    pupilX = gaze.pupil_right_width()
    pupilY = gaze.pupil_right_height()
    
    topLeft = (10,10)
    topRight = (1270, 10)
    bottomLeft = (10, 710)
    bottomRight = (1270,710)
    
    origin = right_pupil
    originX = pupilX
    originY = pupilY
    
    screen = pygame.display.set_mode((500,500)) #x and y are height and width
    pygame.draw.circle(screen, (0, 255, 0), (250,250), 10, 5) 
    #(r, g, b) is color, (x, y) is center, R is radius and w is the thickness of the circle border.
    
    print("STARTING calibration with center coordinates", origin)
        
    '''
    boxTopLeftX = originX - 15
    boxTopLeftY = originY - 15
    boxTopLeft = (boxTopLeftX, boxTopLeftY)

    boxTopRightX = originX + 15
    boxTopRightY = originY - 15
    boxTopRight = (boxTopRightX, boxTopRightY)

    boxBottomLeftX = originX - 15
    boxBottomLeftY = originY + 15
    boxBottomLeft = (boxTopLeftX, boxTopLeftY)

    boxBottomRightX = originX + 15
    boxBottomRightY = originY + 15
    boxBottomRight = (boxBottomRightX, boxBottomRightY)

    print("boxTopLeft: ", boxTopLeft)
    print("boxTopRight: ", boxTopRight)
    print("boxBottomLeft: ", boxBottomLeft)
    print("boxBottomRight: ", boxBottomRight)
    '''

    '''
    if (pupilX <= boxTopLeftX) and (pupilY <= boxTopLeftY):
        print("TOP LEFT")

    if (pupilX >= boxTopRightX) and (pupilY <= boxTopRightY):
        print("TOP RIGHT")
    
    if (pupilX <= boxBottomLeftX) and (pupilY >= boxBottomLeftY):
        print("BOTTOM LEFT")

    if (pupilX >= boxBottomRightX) and (pupilY >= boxBottomRightY):
        print("BOTTOM RIGHT")
    '''
    
    #cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    #cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
