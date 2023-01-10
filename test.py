'''THIS PROGRAM FEEDS LIVE WEBCAM VIDEO INTO A PYGAME WINDOW'''

import pygame.camera
import pygame.image
import sys

import cv2
from gaze_tracking import GazeTracking

pygame.camera.init()

gaze = GazeTracking()
eyecam = cv2.VideoCapture(0)

left_pupil = gaze.pupil_left_coords()
right_pupil = gaze.pupil_right_coords()
pupilX = gaze.pupil_right_width()
pupilY = gaze.pupil_right_height()

overlay = pygame.image.load("overlay.png")		# to be replaced by instruction screen
#overlay = pygame.image.load('overlay.bmp')

cameras = pygame.camera.list_cameras()
webcam = pygame.camera.Camera(cameras[0])

webcam.start()

# grab first frame
ogIMG = webcam.get_image()
scaledIMG = pygame.transform.scale(ogIMG, (500, 500))
img = pygame.transform.flip(scaledIMG, True, False)

WIDTH = img.get_width()
HEIGHT = img.get_height()

screen = pygame.display.set_mode( ( WIDTH, HEIGHT ) )
pygame.display.set_caption("pyGame Camera View")

WHITE =		(255, 255, 255)
GREEN =		(  0, 255,	 0)

topLeft = (50,50)
topRight = (WIDTH-50,50)
bottomLeft = (50,HEIGHT-50)
bottomRight = (WIDTH-50,HEIGHT-50)
center = (WIDTH/2,HEIGHT/2)

circleSize = 20

boxTopLeft = (WIDTH/4, HEIGHT/4)
boxBottomLeft = (WIDTH/4 , HEIGHT/4 * 3)
boxTopRight = (WIDTH/4 * 3, HEIGHT/4)
boxBottomRight = (WIDTH/4 * 3, HEIGHT/4 * 3)

scale = boxTopLeft[0] / topLeft[0]
print(scale)

def draw_box():
	pygame.draw.line(screen, WHITE, boxTopLeft, boxTopRight)
	pygame.draw.line(screen, WHITE, boxTopLeft, boxBottomLeft)
	pygame.draw.line(screen, WHITE, boxBottomLeft, boxBottomRight)
	pygame.draw.line(screen, WHITE, boxTopRight, boxBottomRight)

def draw_circle_TOP_LEFT():
	pygame.draw.circle(screen, GREEN, topLeft, circleSize)
	
def draw_circle_TOP_RIGHT():
	pygame.draw.circle(screen, GREEN, topRight, circleSize)

def draw_circle_BOTTOM_LEFT():
	pygame.draw.circle(screen, GREEN, bottomLeft, circleSize)

def draw_circle_BOTTOM_RIGHT():
	pygame.draw.circle(screen, GREEN, bottomRight, circleSize)

def draw_circle_CENTER():
	pygame.draw.circle(screen, GREEN, center, circleSize)
	
def draw_circle_FIXATION(x,y):
	if x and y:
		pos = (x,y)
		pygame.draw.circle(screen, WHITE, pos, circleSize)
	
def topLeftClicked(x, y):
	if ( (x >= (topLeft[0]-circleSize/2) and x <= (topLeft[0]+circleSize/2)) and (y >= (topLeft[1]-circleSize/2) and y <= (topLeft[1]+circleSize/2))):
		return True
		
def bottomLeftClicked(x, y):
	if ( (x >= (bottomLeft[0]-circleSize/2) and x <= (bottomLeft[0]+circleSize/2)) and (y >= (bottomLeft[1]-circleSize/2) and y <= (bottomLeft[1]+circleSize/2))):
		return True

def topRightClicked(x, y):
	if ( (x >= (topRight[0]-circleSize/2) and x <= (topRight[0]+circleSize/2)) and (y >= (topRight[1]-circleSize/2) and y <= (topRight[1]+circleSize/2))):
		return True
		
def bottomRightClicked(x, y):
	if ( (x >= (bottomRight[0]-circleSize/2) and x <= (bottomRight[0]+circleSize/2)) and (y >= (bottomRight[1]-circleSize/2) and y <= (bottomRight[1]+circleSize/2))):
		return True

def centerClicked(x, y):
	if ( (x >= (center[0]-circleSize/2) and x <= (center[0]+circleSize/2)) and (y >= (center[1]-circleSize/2) and y <= (center[1]+circleSize/2))):
		return True
		
def accuracy(x1, x2, y1, y2):

	x2 = x2 * scale
	y2 = y2 * scale

	x = abs(((x1 - x2) / x1) * 100)
	y = abs(((y1 - y2) / y1) * 100)
	
	total = (x+y)/2
	
	return total

text = ""

centerAccuracy = 0
topLeftAccuracy = 0
topRightAccuracy = 0
bottomLeftAccuracy = 0
bottomRightAccuracy = 0

print("POSITION HEAD WITHIN CENTER SQUARE")
print("TO CALIBRATE: look at and click on each circle one at a time")

while True :

	'''EYE TRACKING CODE'''
	
	# We get a new frame from the webcam
	_, frame = eyecam.read()

	# We send this frame to GazeTracking to analyze it
	gaze.refresh(frame)

	frame = gaze.annotated_frame()
	
	pygame.display.update()

	# include these values again so they are continuously updated
	left_pupil = gaze.pupil_left_coords()
	right_pupil = gaze.pupil_right_coords() # these are the coordinates ON THE SCREEN
	
	# scale to get info about coordinates relative to where person is looking
	pupilX = gaze.pupil_right_width() 
	pupilY = gaze.pupil_right_height() 
	
	
	draw_circle_FIXATION(pupilX, pupilY)
	
	mouseX, mouseY = pygame.mouse.get_pos()
	
	#cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
	
	for e in pygame.event.get() :
	
		if e.type == pygame.MOUSEBUTTONDOWN:
			
			if centerClicked(mouseX, mouseY):
				origin = right_pupil
				originX = pupilX * scale
				originY = pupilY * scale
				centerAccuracy = accuracy(originX, center[0], originY, center[1])

				print("calibrating center")
				break
		
			if topLeftClicked(mouseX, mouseY):
				topLeftX = pupilX * scale
				topLeftY = pupilY * scale
				topLeftAccuracy = accuracy(topLeftX, topLeft[0], topLeftY, topLeft[1])
			
				print("calibrating top left")
				break
		
			if topRightClicked(mouseX, mouseY):
				topRightX = pupilX * scale
				topRightY = pupilY * scale
				topRightAccuracy = accuracy(topRightX, topRight[0], topRightY, topRight[1])
			
				print("calibrating top right")
				break
		
			if bottomLeftClicked(mouseX, mouseY):
				bottomLeftX = pupilX * scale
				bottomLeftY = pupilY * scale
				bottomLeftAccuracy = accuracy(bottomLeftX, bottomLeft[0], bottomLeftY, bottomLeft[1])
			
				print("calibrating bottom left")
				break
		
			if bottomRightClicked(mouseX, mouseY):
				bottomRightX = pupilX * scale
				bottomRightY = pupilY * scale
				bottomRightAccuracy = accuracy(bottomRightX, bottomRight[0], bottomRightY, bottomRight[1])
			
				print("calibrating bottom right")
				break
		
		#print("finished/not calibrating")
		
		'''SENSORS: WHICH CIRCLE HAS THE PUPIL'S ATTENTION?'''
  
		'''
		# TOP LEFT SENSOR
		if ( pupilX >= topLeft[0] and pupilY >= topLeft[1] ):
			print("DETECTED: top left")

		# TOP RIGHT SENSOR
		while ( pupilX >= topRight[0] and pupilY <= topRight[1] ):
			print("DETECTED: top right")

		# BOTTOM LEFT SENSOR
		while ( pupilX <= bottomLeft[0] and pupilY >= bottomLeft[1] ):
			print("DETECTED: bottom left")

		# BOTTOM RIGHT SENSOR
		while ( pupilX <= bottomLeft[0] and pupilY >= bottomRight[1] ):
			print("DETECTED: bottom right")

		# CENTER SENSOR
		while ( pupilX >= center[0] and pupilY <= center[1] ):
			print("DETECTED: center")
		'''
	
		if e.type == pygame.QUIT :
			print("ACCURACY:")
			print("center: ", centerAccuracy)
			print("topLeft: ", topLeftAccuracy)
			print("topRight: ", topRightAccuracy)
			print("bottomLeft: ", bottomLeftAccuracy)
			print("bottomRight: ", bottomRightAccuracy)

			acc = (topLeftAccuracy + topRightAccuracy + bottomLeftAccuracy + bottomRightAccuracy + centerAccuracy) / 5
			print("TOTAL ACCURACY: ", acc)
			
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
	draw_circle_CENTER() 
	draw_box()
	pygame.display.update()
	

	'''ONSCREEN TEXT'''

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

	
	#cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
	#cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

	#cv2.imshow("Demo", frame)

	if cv2.waitKey(1) == 27:
		break


eyecam.release()
cv2.destroyAllWindows()



