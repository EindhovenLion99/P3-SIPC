# Librerias

import numpy as np
import cv2 
import math
import imutils
from variables_dibujo import *

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

  # *************************** Seccion de dibujo ***********************************************

  # Cambiamos la deteccion de colores a formato HSV
  frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Creamos una matriz de ceros igual a frames, para luego dibujar en aux
  if aux is None: aux = np.zeros(frame.shape, dtype = np.uint8)

  # Rectangulos de seleccion de colores
  cv2.rectangle(frame,(0,0),(50,50), colorAmarillo, 3)
  cv2.rectangle(frame,(50,0),(100,50), colorAzul, 3)
  cv2.rectangle(frame,(100,0),(150,50), colorVerde, 3)

  # Rectangulo para limpiar pantalla
  cv2.rectangle(frame,(300, 0), (400, 50), LimpiarPantalla, 1)
  cv2.putText(frame, 'Borrar', (320, 20), 2, 0.6, LimpiarPantalla, 1, cv2.LINE_AA)

  # Deteccion del color Azul
  maskAzul = cv2.inRange(frameHSV, AzulClaro, AzulFuerte)

  # Transformaciones morfologicas para mejorar la deteccion del azul
  maskAzul = cv2.erode(maskAzul, None, iterations=1)
  maskAzul = cv2.dilate(maskAzul, None, iterations=2)
  maskAzul = cv2.medianBlur(maskAzul, 13)

  # Deteccion del contorno del color azul en la camara
  cnts, _ = cv2.findContours(maskAzul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Seleccionamos el contorno mas grande
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

  # Frame donde se va a mostrar los colores azules que ve la camara
  maskAzulV = cv2.bitwise_and(frame, frame, mask = maskAzul)

  for c in cnts:
    # Asignamos el area de la tapa del boli detectada
    area = cv2.contourArea(c)
    if area > 1000:
      # Asigna un rectangulo al area de la tapa del boli azul
      x, y2, w, h = cv2.boundingRect(c)

      # Coordenada central de la punta de la tapa del boli
      x2 = x + w//2 

      if x1 is not None:
        if 0 < x2 < 50 and 0 < y2 < 50:
          # Si el boli esta por encima del recuadro amarillo, este color se selecciona
          color = colorAmarillo
        if 50 < x2 < 100 and 0 < y2 < 50:
          color = colorAzul
        if 100 < x2 < 150 and 0 < y2 < 50:
          color = colorVerde

        # Borra todo lo escrito en pantalla si se situa la punta del boli en el rectangulo
        if 300 < x2 < 400 and 0 < y2 < 50:
          cv2.rectangle(frame,(300, 0), (400, 50), LimpiarPantalla, 2) # Destacamos el rectangulo en pantalla
          cv2.putText(frame, 'Borrar', (320, 20), 2, 0.6, LimpiarPantalla, 2, cv2.LINE_AA) # Destacamos las letras en pantalla
          # Borramos el contenide de aux
          aux = np.zeros(frame.shape, dtype = np.uint8)
        
        # Si el boli se situa en la parte superior del frame, deja de pintar
        if 0 < y2 < 60 or 0 < y1 < 60:
          aux = aux
        
        # Pintamos en pantalla con las opciones anteriores
        else:
          aux = cv2.line(aux, (x1,y1), (x2, y2), color, 4)
      
      # Destacamos el boli con un circulito azul en su punta
      cv2.circle(frame, (x2,y2), 3, color, 3)
      # Ajustamos las coordenadas para el dibujo
      x1 = x2
      y1 = y2
    else:
      x1 = None
      y1 = None

  # *********************************************************************************

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
          #cv2.putText(roi, '{}'.format(fingers), tuple(cnt), 1, 1, color_fingers, 1, cv2.LINE_AA)

      for i in range(len(beginning)):
        fingers = fingers + 1
        #cv2.putText(roi, '{}'.format(fingers), tuple(beginning[i]), 1, 1, color_fingers, 1, cv2.LINE_AA)
        if i == len(beginning) - 1:
          fingers = fingers + 1
          #cv2.putText(roi, '{}'.format(fingers), tuple(ending[i]), 1, 1, color_fingers, 1, cv2.LINE_AA)
      cv2.putText(frame, '{}'.format(fingers), (390, 45), 1, 4, (color_fingers), 2, cv2.LINE_AA)
      rect = cv2.boundingRect(cnt)
      pt1_ = (rect[0],rect[1])
      pt2_ = (rect[0]+rect[2],rect[1]+rect[3])
      cv2.rectangle(roi,pt1_,pt2_,(0,0,255),3)


  cv2.imshow('FgMask', fgMask)
  cv2.imshow('frame', frame)
  cv2.imshow('ROI', roi)
  cv2.imshow('Dibujo', aux) # Zona de Dibujo
  cv2.imshow('Azul', maskAzulV) # Deteccion del azul
  # cv2.imshow('prueba', opening)

  keyboard = cv2.waitKey(10)
  if keyboard & 0xFF == ord('d'):
    lr = 0

  keyboard = cv2.waitKey(10)
  if keyboard & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()