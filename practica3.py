import numpy as np
import cv2 

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

flag = None

if not cap.isOpened:
  print("Unable to open the cam")
  exit(0)
pt1 = (400, 100)
pt2 = (600, 300)

frame_width  = int(cap.get(3))
frame_height = int(cap.get(4))

lr = -1
while True:
  ret,frame=cap.read()
  if not ret:
    exit(0)

  frame = cv2.flip(frame,1)
  roi = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:].copy()
  cv2.rectangle(frame,pt1,pt2,(173, 255, 51))
  fgMask = backSub.apply(roi,learningRate = lr)

  #Removing noise
  #kernel = np.ones((5,5),np.uint8)
  #opening = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
  contours, hierarchy = cv2.findContours(fgMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
  if len(contours) > 0:
    max = -1
    #index = None
    for i, cnt in enumerate(contours):
      auxMax = len(cnt)
      if (max < auxMax):
        max = auxMax
        index = i
  
    cv2.drawContours(roi, contours, index, (0,255,0))
    hull = cv2.convexHull(contours[index])
    cv2.drawContours(roi, [hull], 0, (255,0,0), 3)

  cv2.imshow('FgMask', fgMask)
  cv2.imshow('frame', frame)
  cv2.imshow('ROI', roi)
  #cv2.imshow('prueba', opening)

  keyboard = cv2.waitKey(10)
  if keyboard & 0xFF == ord('d'):
    lr = 0

  keyboard = cv2.waitKey(10)
  if keyboard & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
