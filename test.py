'''THIS PROGRAM FEEDS LIVE WEBCAM VIDEO INTO A PYGAME WINDOW'''

import pygame.camera
import pygame.image
import sys

import cv2
from gaze_tracking import GazeTracking

pygame.camera.init()

gaze = GazeTracking()
eyecam = cv2.VideoCapture(0)


overlay = pygame.image.load("overlay.png")      # to be replaced by instruction screen
#overlay = pygame.image.load('overlay.bmp')

cameras = pygame.camera.list_cameras()

print ("Using camera %s ..." % cameras[0])

webcam = pygame.camera.Camera(cameras[0])

webcam.start()

# grab first frame
img = webcam.get_image()

WIDTH = img.get_width()
HEIGHT = img.get_height()

screen = pygame.display.set_mode( ( WIDTH, HEIGHT ) )
pygame.display.set_caption("pyGame Camera View")

WHITE =     (255, 255, 255)
GREEN =     (  0, 255,   0)


def draw_circle_TOP_LEFT():
    pos=(50,50)
    pygame.draw.circle(screen, GREEN, pos, 20)
    
def draw_circle_TOP_RIGHT():
    pos=(WIDTH-50,50)
    pygame.draw.circle(screen, GREEN, pos, 20)

def draw_circle_BOTTOM_LEFT():
    pos=(50,HEIGHT-50)
    pygame.draw.circle(screen, GREEN, pos, 20)

def draw_circle_BOTTOM_RIGHT():
    pos=(WIDTH-50,HEIGHT-50)
    pygame.draw.circle(screen, GREEN, pos, 20)

while True :
    for e in pygame.event.get() :
    
        if e.type == pygame.QUIT :
            sys.exit()
    
    # draw frame
    screen.blit(img, (0,0))
    
    pygame.display.flip()
    # grab next frame    
    img = webcam.get_image()
    
    # draw the circles
    draw_circle_TOP_LEFT()
    draw_circle_TOP_RIGHT()
    draw_circle_BOTTOM_LEFT()
    draw_circle_BOTTOM_RIGHT()
    
    pygame.display.update()
    
    '''EYE TRACKING CODE'''
    
    # We get a new frame from the webcam
    _, frame = eyecam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    
    pygame.display.update()

    
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

    #cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
    
    

eyecam.release()
cv2.destroyAllWindows()

