import numpy as np
import cv2


cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

if not cap.isOpened:
    print ("Unable to open cam")
    exit(0)

while (True):
	ret,frame=cap.read()
	if not ret:
		exit(0)
	fgMask = backSub.apply(frame)
	cv2.imshow('frame',frame)
	cv2.imshow('Foreground Mask',fgMask)

	keyboard = cv2.waitKey(1)
	if keyboard & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

