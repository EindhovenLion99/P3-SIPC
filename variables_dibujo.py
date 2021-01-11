# Librerias

import cv2
import numpy as np

# Rango de colores azul a detectar por la camara
# Formato HSV [H, S, V]

AzulClaro = np.array([100, 100, 20], np.uint8)
AzulFuerte = np.array([125, 255, 255], np.uint8)

# Colores y grosor, formato [B, G, R]

colorAmarillo = (90, 225, 250)
colorVerde = (0, 255, 0)
colorAzul = (255, 0, 0)

LimpiarPantalla = (0, 0, 0) # Negro

grosorLinea = 5

# Color y grosor por defecto para el boli

color = colorAzul
grosor = 3

# Variables de la linea para pintar

x1 = None
y1 = None
aux = None

