import numpy as np
import cv2 

cap = cv2.VideoCapture(0)

if not cap.isOpened:
  print("Unable to open the cam")
  exit(0)
pt1 = (400, 100)
pt2 = (600, 300)

frame_width  = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while(True):
  ret,frame=cap.read()
  if not ret:
    exit(0)

  frame = cv2.flip(frame,1)

  roi = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:].copy()

  cv2.rectangle(frame,pt1,pt2,(173, 255, 51))
  cv2.imshow('frame',frame)
  cv2.imshow('ROI',roi)
  out.write(frame)
  keyboard = cv2.waitKey(40)
  if keyboard & 0xFF == ord('q'):
    break

cap.release()
out.release()
cv2.destroyAllWindows()
