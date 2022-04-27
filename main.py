
# Steering Wheel using OpenCV
# Working condition

import numpy as np
import math
import cv2
import time
from Keys import PressKey, ReleaseKey ,W,S,A,D


currentKey = []

#CAPTURE WEBCAM INPUT
cap = cv2.VideoCapture(0)

# HEIGHT & WEIGHT of Webcam Window
We = int(cap.get(3))
He = int(cap.get(4))


# COLOR TO BE DETECTED ( HSV FOR GREEN )
lower = np.array([25,52,72])
upper = np.array([102,255,255])


# FIND INTERSECTION OF TWO LINES
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])


    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


# TO FIND GRADIENT OF 2 POINTS
def gradient(pt1 , pt2):
	
	try:
		ans = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

	except ZeroDivisionError:
		ans = 0

	return ans


# TO FIND ANGLE MADE BY 2 LINES
def getAngle(a,b,c):
	
	m1 = gradient(a,b)
	m2 = gradient(a,c)
	angR = math.atan((m2-m1)/(1+(m2*m1)))
	angD = round(math.degrees(angR))

	return angD

def right():
	PressKey(D)
	ReleaseKey(A)
	ReleaseKey(W)

def left():
	PressKey(A)
	ReleaseKey(D)
	ReleaseKey(W)

def straight():
	PressKey(W)
	ReleaseKey(A)
	ReleaseKey(D)


# -------------------- MAIN -----------------------------
time.sleep(4)
while True:

	is_turning = False

	_ , frame = cap.read()

	# DRAW STRAIGHT BLACK LINE 
	X_cen = int(We / 2)
	Y_cen = int(He / 2)
	frame = cv2.line(frame , (X_cen , 0) , ([X_cen , He]) , (0,0,0) , 3)

	# CONVERT BGR to HSV
	hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)

	# MASK COLOR 
	g_mask = cv2.inRange(hsv , lower , upper)

	# FIND ALL CONTOURS
	conts , hir = cv2.findContours(g_mask , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)


	if len(conts) != 0:
		con_point = [] # to store center of each contour
		for c in conts:
			if cv2.contourArea(c) > 5000:

				# DRAW RECHTANGLE AROUND CONTOUR
				x,y,w,h = cv2.boundingRect(c)
				cv2.rectangle(frame , (x,y) , (x+w , y+h), (0,0,255),3)

				# CENTER POINTS
				M = cv2.moments(c)
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
				
				#APPEND CENTER POINT
				con_point.append((cX , cY))
				print(cX)
				#DRAW CENTER POINT
				cv2.circle(frame , (cX , cY), 7, (255,255,255), -1)

		#JOIN CENTERS
		if len(con_point) >= 2 :

			#LINE that JOINS CENTER
			cv2.line(frame , con_point[0] , con_point[1] , (255,255,255), 1)

			# 2 LINES POINTS
			line1 = (list(con_point[0]) , list(con_point[1]))
			line2 = ([X_cen , 0] , [X_cen , He])

			# INTERSECTION POINT 
			inter = line_intersection(line1, line2)
			inter = (int(inter[0]) , int(inter[1]))
			cv2.circle(frame , inter, 7, (255,255,255), -1)

			# ANGLE FINDING
			angle = getAngle(inter , (X_cen , 0) , con_point[1] )
			print(angle)
			
			# AUTOMATIC MOVE FORWARD CODE
			PressKey(W)

			# LOGIC TO TURN ON BASIS OF ANGLE
			if angle > 10:
				PressKey(A)
				is_turning = True
				currentKey.append(A)

			elif angle < -10:
				PressKey(D)
				is_turning = True
				currentKey.append(D)

			if not is_turning and len(currentKey) != 0:
				for current in currentKey:
					ReleaseKey(current)
				currentKey = list()

				
	#SHOW WINDOW
	cv2.imshow('NEW' , frame)
	
	# USE 'q' to EXIT
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()