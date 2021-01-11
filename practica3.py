# Librerias
import numpy as np
import cv2 
import math
import imutils
from variables_dibujo import *

 # Variable que contiene la imagen que se recoge desde la cámara por defecto del ordenador
cap = cv2.VideoCapture(0)    

# Se hace uso de un método para hacer la substracción de fondo que obtiene una imagen binaria que reconoce las sombras
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

# Color para la visualización del total de dedos levantados
color_fingers = (0, 255, 255) 

# Se verifica que se ha abierto correctamente la capturadora de pantalla
if not cap.isOpened:
  print("Unable to open the cam")
  exit(0)

# Se define la esquina superior izquierda y la esquina inferior derecha de nuestra región de interés
pt1 = (400, 100)
pt2 = (600, 300)

frame_width  = int(cap.get(3))
frame_height = int(cap.get(4))

# Método que permite obtener el ángulo formado entre dos dedos, a partir del triángulo que se forma por los 
#   segmentos del punto inicial al punto más alejado, del punto más alejado al punto final y del punto final 
#   al inicial
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

# Se define el valor por defecto del learning rate
lr = -1
while True:
  # Siempre que se pueda ver cada frame correctamente se ejecutará el programa
  ret,frame=cap.read()
  if not ret:
    exit(0)

  # Se voltea la imagen reconocida por pantalla para obtener un efecto espejo
  frame = cv2.flip(frame,1)

  # Determinamos la región de interés
  roi = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:].copy()
  cv2.rectangle(frame,pt1,pt2,(173, 255, 51))

  # Aplicamos la substracción de fondo en la región de interés, junto con su learning rate
  fgMask = backSub.apply(roi,learningRate = lr)

  # *************************** Sección de dibujo ***********************************************

  # Cambiamos la detección de colores a formato HSV
  frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  # Creamos una matriz de ceros igual a frames, para luego dibujar en aux
  if aux is None: aux = np.zeros(frame.shape, dtype = np.uint8)

  # Rectángulos de seleccion de colores
  cv2.rectangle(frame,(0,0),(50,50), colorAmarillo, 3)
  cv2.rectangle(frame,(50,0),(100,50), colorAzul, 3)
  cv2.rectangle(frame,(100,0),(150,50), colorVerde, 3)

  # Rectángulo para limpiar pantalla
  cv2.rectangle(frame,(300, 0), (400, 50), LimpiarPantalla, 1)
  cv2.putText(frame, 'Borrar', (320, 20), 2, 0.6, LimpiarPantalla, 1, cv2.LINE_AA)

  # Detección del color Azul
  maskAzul = cv2.inRange(frameHSV, AzulClaro, AzulFuerte)

  # Transformaciones morfológicas para mejorar la detección del azul
  maskAzul = cv2.erode(maskAzul, None, iterations=1)
  maskAzul = cv2.dilate(maskAzul, None, iterations=2)
  maskAzul = cv2.medianBlur(maskAzul, 13)

  # Detección del contorno del color azul en la cámara
  cnts, _ = cv2.findContours(maskAzul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Seleccionamos el contorno mas grande
  cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

  # Frame donde se va a mostrar los colores azules que ve la cámara
  maskAzulV = cv2.bitwise_and(frame, frame, mask = maskAzul)

  for c in cnts:
    # Asignamos el área de la tapa del boli detectada
    area = cv2.contourArea(c)
    if area > 1000:
      # Asigna un rectángulo al area de la tapa del boli azul
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
          # Borramos el contenido de aux
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

  # Variables y transformaciones morfológicas de apertura que permiten disminuir el ruido de la imagen 
  kernel = np.ones((5,5),np.uint8)
  opening = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

  # Se encuentran todos los posibles contornos de la imagen
  contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

  # Siempre que haya puntos de contornos en el vector, se buscará el contorno más largo y además se almacenará el índice del mismo
  if len(contours) > 0:
    max = -1
    for i, cnt in enumerate(contours):
      auxMax = len(cnt)
      if (max < auxMax):
        max = auxMax
        index = i

      # Encontrar centro del contorno
      M = cv2.moments(cnt)
      if M["m00"] == 0: M["m00"] = 1
      x = int(M["m10"]/M["m00"])
      y = int(M["m01"]/M["m00"])
      cv2.circle(roi, tuple([x, y]), 5, (0, 255, 0), -1) # DIbujar un circulo en el centro del contorno

      cv2.drawContours(roi, contours, index, (0,255,0))

    cnt = contours[index]

    # Definición de la malla convexa
    hull = cv2.convexHull(cnt,returnPoints = False)

    # Encontrando los defectos de convexidad 
    defects = cv2.convexityDefects(cnt,hull)

    # Siempre que se hayan encontrado defectos de convexidad...
    if defects is not None:
      
      beginning = [] # Lista que almacena los puntos iniciales de los defectos de convexidad
      ending = []    # Lista que almacena los puntos finales de los defectos de convexidad
      fingers = 0    # Contador para reconocer la cantidad de dedos levantados
       
      for i in range(len(defects)):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        depth = d/256.0
        ang = angle(start,end,far)

        # Se descartan defectos convexos dependiendo de su distancia entre start, end y d, además del ángulo formado ang
        if np.linalg.norm(cnt[s][0] - cnt[e][0]) > 20 and d > 12000 and ang < 75:
          beginning.append(start)  
          ending.append(end)
          cv2.line(roi,start,end,[255,0,0],2)
          cv2.circle(roi,far,5,[0,0,255],-1)  


      # En caso de no haber almacenado puntos de inicio
      if len(beginning) == 0:
        # Cuando no se ha levantado ningún dedo
        minY = np.linalg.norm(cnt - [x, y])

        # Cuando hay un dedo levantado
        if minY >= 850:
          fingers = fingers + 1

      # Contador de dedos levantados 
      for i in range(len(beginning)):
        fingers = fingers + 1
        if i == len(beginning) - 1:
          fingers = fingers + 1

      # Visualización por pantalla del total de dedos levantados a tiempo real
      cv2.putText(frame, '{}'.format(fingers), (390, 45), 1, 4, (color_fingers), 2, cv2.LINE_AA)

      # Creación del bounding rect
      rect = cv2.boundingRect(cnt)
      pt1_ = (rect[0],rect[1])
      pt2_ = (rect[0]+rect[2],rect[1]+rect[3])
      cv2.rectangle(roi,pt1_,pt2_,(0,0,255),3)


  cv2.imshow('Dibujo', aux)     # Zona de Dibujo
  cv2.imshow('Azul', maskAzulV) # Deteccion del azul
  cv2.imshow('frame', frame)    # Ventana principal con diferentes opciones y región de interés
  cv2.imshow('ROI', roi)        # Región de interés
  cv2.imshow('FgMask', fgMask)  # Región de interés con la substracción de fondo aplicada


  # Se está a la espera de pulsar la tecla d para cambiar el valor del learning rate y así dejar de aprender
  keyboard = cv2.waitKey(10)
  if keyboard & 0xFF == ord('d'):
    lr = 0

  # Se está a la espera de pulsar la tecla q para dar por finalizada la ejecución del programa
  keyboard = cv2.waitKey(10)
  if keyboard & 0xFF == ord('q'):
    break

cap.release()             # Se libera la capturación de pantalla
cv2.destroyAllWindows()   # Se eliminan las ventanas creadas