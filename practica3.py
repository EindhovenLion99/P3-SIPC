import numpy as np
import cv2 
import math
import imutils 

cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
color_fingers = (0, 255, 255)

if not cap.isOpened:
  print("Unable to open the cam")
  exit(0)
pt1 = (400, 100)
pt2 = (600, 300)

frame_width  = int(cap.get(3))
frame_height = int(cap.get(4))


def angle(s,e,f):
    v1 = [s[0]-f[0],s[1]-f[1]]
    v2 = [e[0]-f[0],e[1]-f[1]]
    ang1 = math.atan2(v1[1],v1[0])
    ang2 = math.atan2(v2[1],v2[0])
    ang = ang1 - ang2
    if (ang > np.pi):
        ang -= 2*np.pi
    if (ang < -np.pi):
        ang += 2*np.pi
    return ang*180/np.pi

lr = -1
while True:
  ret,frame=cap.read()
  if not ret:
    exit(0)

  frame = cv2.flip(frame,1)
  roi = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:].copy()
  cv2.rectangle(frame,pt1,pt2,(173, 255, 51))
  fgMask = backSub.apply(roi,learningRate = lr)

  kernel = np.ones((5,5),np.uint8)
  opening = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

  contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
  if len(contours) > 0:
    max = -1
    for i, cnt in enumerate(contours):
      auxMax = len(cnt)
      if (max < auxMax):
        max = auxMax
        index = i

      #Encontrar centro
      M = cv2.moments(cnt)
      if M["m00"] == 0: M["m00"] = 1
      x = int(M["m10"]/M["m00"])
      y = int(M["m01"]/M["m00"])
      cv2.circle(roi, tuple([x, y]), 5, (0, 255, 0), -1)

      cv2.drawContours(roi, contours, index, (0,255,0))
      #hull = cv2.convexHull(contours[index])
      #cv2.drawContours(roi, [hull], 0, (255,0,0), 3)
    cnt = contours[index]
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)

    if defects is not None:
      beginning = []
      ending = []
      fingers = 0
       
      for i in range(len(defects)):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        depth = d/256.0
        #print(depth)
        ang = angle(start,end,far)

        if np.linalg.norm(cnt[s][0] - cnt[e][0]) > 20 and d > 12000 and ang < 75:
          beginning.append(start)
          ending.append(end)
          cv2.line(roi,start,end,[255,0,0],2)
          cv2.circle(roi,far,5,[0,0,255],-1)  


      #Cuando se levanta un solo dedo
      if len(beginning) == 0:
        minY = np.linalg.norm(cnt - [x, y])
        if minY >= 850:
          fingers = fingers + 1
          #cv2.putText(roi, '{}'.format(fingers), (100,100), 1, 1, color_fingers, 1, cv2.LINE_AA)

      for i in range(len(beginning)):
        fingers = fingers + 1
        cv2.putText(roi, '{}'.format(fingers), tuple(beginning[i]), 1, 1, color_fingers, 1, cv2.LINE_AA)
        if i == len(beginning) - 1:
          fingers = fingers + 1
          cv2.putText(roi, '{}'.format(fingers), tuple(ending[i]), 1, 1, color_fingers, 1, cv2.LINE_AA)
      cv2.putText(frame, '{}'.format(fingers), (390, 45), 1, 4, (color_fingers), 2, cv2.LINE_AA)
      rect = cv2.boundingRect(cnt)
      pt1_ = (rect[0],rect[1])
      pt2_ = (rect[0]+rect[2],rect[1]+rect[3])
      cv2.rectangle(roi,pt1_,pt2_,(0,0,255),3)


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
