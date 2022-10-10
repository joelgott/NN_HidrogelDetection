# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math
import random
import yaml
import os
#from google.colab.patches import cv2_imshow

def mindist(x,y):
  min = float('inf')
  if len(x):
    for i in x:
      dist = math.sqrt((i.object_x_center-y[0])**2+(i.object_y_center-y[1])**2)
      if dist < min:
        min = dist  
  return min

class yolo_object:
  def __init__(self, yolo_class, object_x_center, object_y_center,object_x_width, object_y_width):
    self.yolo_class = yolo_class
    self.object_x_center = object_x_center
    self.object_y_center = object_y_center
    self.object_x_width = object_x_width
    self.object_y_width = object_y_width

silovacio = cv2.imread('silovacio.png', cv2.IMREAD_UNCHANGED) # b = bola bs = bola sombreada
b1 = cv2.imread('bola1.png', cv2.IMREAD_UNCHANGED)
b2 = cv2.imread('bola2.png', cv2.IMREAD_UNCHANGED)
b3 = cv2.imread('bola3.png', cv2.IMREAD_UNCHANGED)
b4 = cv2.imread('bola4.png', cv2.IMREAD_UNCHANGED)
b5 = cv2.imread('bola5.png', cv2.IMREAD_UNCHANGED)
b6 = cv2.imread('bola6.png', cv2.IMREAD_UNCHANGED)
b7 = cv2.imread('bola7.png', cv2.IMREAD_UNCHANGED)
b8 = cv2.imread('bola8.png', cv2.IMREAD_UNCHANGED)
b9 = cv2.imread('bola9.png', cv2.IMREAD_UNCHANGED)
b10 = cv2.imread('bola10.png', cv2.IMREAD_UNCHANGED)
b11 = cv2.imread('bola11.png', cv2.IMREAD_UNCHANGED)
b12 = cv2.imread('bola12.png', cv2.IMREAD_UNCHANGED)

bs1 = cv2.imread('bolasombreada1.png', cv2.IMREAD_UNCHANGED)
bs2 = cv2.imread('bolasombreada2.png', cv2.IMREAD_UNCHANGED)
bs3 = cv2.imread('bolasombreada3.png', cv2.IMREAD_UNCHANGED)
bs4 = cv2.imread('bolasombreada4.png', cv2.IMREAD_UNCHANGED)
bs5 = cv2.imread('bolasombreada5.png', cv2.IMREAD_UNCHANGED)
bs6 = cv2.imread('bolasombreada6.png', cv2.IMREAD_UNCHANGED)
bs7 = cv2.imread('bolasombreada7.png', cv2.IMREAD_UNCHANGED)
bs8 = cv2.imread('bolasombreada8.png', cv2.IMREAD_UNCHANGED)

bolas = [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12]
bolas_classes = [0,1,0,0,0,0,1,0,0,0,0,1]

bolas_sombreadas = [bs1,bs2,bs3,bs4,bs5,bs6,bs7,bs8]
bolas_sombreadas_classes = [0,0,0,1,1,0,0,0]

image_x = silovacio.shape[1]
image_y = silovacio.shape[0]

proximity = 15/image_x
min_balls = 75
max_balls = 200
train_amount = 300
valid_amount = 50

#os.rmdir('train')
#os.rmdir('valid')
#os.rmdir('test')
os.mkdir('train')
os.chdir('train')
os.mkdir('labels')
os.mkdir('images')
os.chdir("../")
os.mkdir('valid')
os.chdir('valid')
os.mkdir('labels')
os.mkdir('images')
os.chdir("../")
os.mkdir('test')
os.chdir('test')
os.mkdir('labels')
os.mkdir('images')
os.chdir("../")

y_max = 920
y_min = 30

contour = np.array([[475,y_max],[550,y_max],[740,y_min],[270,y_min]])
contour = contour.reshape((-1,1,2)).astype(np.int32)

contoursombra = np.array([[475,y_max],[550,y_max],[595,770],[430,770]])
contoursombra = contoursombra.reshape((-1,1,2)).astype(np.int32)

for i in range(train_amount):

  silo = silovacio.copy()

  positions = []

  N = 0
  balls_amount = random.randrange(min_balls,max_balls)
  while N < balls_amount:

    x_center = random.randrange(silo.shape[1]//4, silo.shape[1]*3//4)
    y_center = random.randrange(y_min, y_max)


    if((cv2.pointPolygonTest(contour,(x_center,y_center),True) > 0) and (mindist(positions,(x_center/image_x,y_center/image_y)) > proximity)):
      if(cv2.pointPolygonTest(contoursombra,(x_center,y_center),True) < 0):
        index = random.randrange(len(bolas))
        bola = bolas[index]
        bola_size = bola.shape[0]
        bola_class = bolas_classes[index]
      if(cv2.pointPolygonTest(contoursombra,(x_center,y_center),True) >= 0):
        index = random.randrange(len(bolas_sombreadas))
        bola = bolas_sombreadas[index]
        bola_size = bola.shape[0]
        bola_class = bolas_sombreadas_classes[index]

      x_offset = x_center - bola_size//2
      y_offset = y_center - bola_size//2
      x_end = x_offset + bola_size
      y_end = y_offset + bola_size


      exterior = cv2.bitwise_or(silo[y_offset:y_end,x_offset:x_end,0:3],silo[y_offset:y_end,x_offset:x_end,0:3],mask = cv2.bitwise_not(bola[:,:,3]))
      final = cv2.add(exterior,bola[:,:,0:3])

      silo[y_offset:y_end,x_offset:x_end,0:3] = final
      #cv2.rectangle(silo, (x_offset,y_offset), (x_end,y_end), (0, 0, 255, 255), 2)

      bola_yolo = yolo_object(bola_class,x_center/image_x,y_center/image_y,bola_size/image_x,bola_size/image_y)
      positions.append(bola_yolo)
      N += 1

  name = "train"
  os.chdir('train/images')
  cv2.imwrite(name+str(i)+'.jpg', silo)
  os.chdir('../labels')
  with open(name+str(i)+'.txt', 'w') as f:
      for obj in positions:
        f.write(str(obj.yolo_class)+' '+str(obj.object_x_center)+' '+str(obj.object_y_center)+' '+str(obj.object_x_width)+' '+str(obj.object_y_width)+' '+"\n")
  os.chdir('../../')

for i in range(valid_amount):

  silo = silovacio.copy()

  positions = []

  N = 0
  balls_amount = random.randrange(min_balls,max_balls)
  while N < balls_amount:

    x_center = random.randrange(silo.shape[1]//4, silo.shape[1]*3//4)
    y_center = random.randrange(y_min, y_max)


    if((cv2.pointPolygonTest(contour,(x_center,y_center),True) > 0) and (mindist(positions,(x_center/image_x,y_center/image_y)) > proximity)):
      if(cv2.pointPolygonTest(contoursombra,(x_center,y_center),True) < 0):
        index = random.randrange(len(bolas))
        bola = bolas[index]
        bola_size = bola.shape[0]
        bola_class = bolas_classes[index]
      if(cv2.pointPolygonTest(contoursombra,(x_center,y_center),True) >= 0):
        index = random.randrange(len(bolas_sombreadas))
        bola = bolas_sombreadas[index]
        bola_size = bola.shape[0]
        bola_class = bolas_sombreadas_classes[index]

      x_offset = x_center - bola_size//2
      y_offset = y_center - bola_size//2
      x_end = x_offset + bola_size
      y_end = y_offset + bola_size


      exterior = cv2.bitwise_or(silo[y_offset:y_end,x_offset:x_end,0:3],silo[y_offset:y_end,x_offset:x_end,0:3],mask = cv2.bitwise_not(bola[:,:,3]))
      final = cv2.add(exterior,bola[:,:,0:3])

      silo[y_offset:y_end,x_offset:x_end,0:3] = final
      #cv2.rectangle(silo, (x_offset,y_offset), (x_end,y_end), (0, 0, 255, 255), 2)

      bola_yolo = yolo_object(bola_class,x_center/image_x,y_center/image_y,bola_size/image_x,bola_size/image_y)
      positions.append(bola_yolo)
      N += 1

  name = "valid"
  os.chdir('valid/images')
  cv2.imwrite(name+str(i)+'.jpg', silo)
  os.chdir('../labels')
  with open(name+str(i)+'.txt', 'w') as f:
      for obj in positions:
        f.write(str(obj.yolo_class)+' '+str(obj.object_x_center)+' '+str(obj.object_y_center)+' '+str(obj.object_x_width)+' '+str(obj.object_y_width)+' '+"\n")
  os.chdir('../../')

