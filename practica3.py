import numpy as np
import cv2 


cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
auxBackSub = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

flag = None

if not cap.isOpened:
  print("Unable to open the cam")
  exit(0)
pt1 = (400, 100)
pt2 = (600, 300)

frame_width  = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
  ret,frame=cap.read()
  if not ret:
    exit(0)

  frame = cv2.flip(frame,1)
  roi = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:].copy()
  cv2.rectangle(frame,pt1,pt2,(173, 255, 51))
  fgMask = backSub.apply(roi)

  if flag is not None:
    fgMask = auxBackSub.apply(roi,learningRate = 0)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ret,bw = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(roi, contours, -1, (0,255,0))
    #hull = cv2.convexHull(contours[0])
    #cv2.drawContours(roi, [hull], 0, (255,0,0),3)
   
  cv2.imshow('FgMask', fgMask)
  cv2.imshow('frame', frame)
  cv2.imshow('ROI', roi)
  out.write(frame)

  keyboard = cv2.waitKey(10)
  if keyboard & 0xFF == ord('d'):
    flag = 1

  keyboard = cv2.waitKey(10)
  if keyboard & 0xFF == ord('q'):
    break

cap.release()
out.release()
cv2.destroyAllWindows()
