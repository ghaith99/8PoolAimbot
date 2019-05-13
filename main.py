import cv2
import numpy as np
import matplotlib.pyplot as plt


def nothing(x):
    pass


def canny(image, threshold1 = 33,threshold2 = 0) :
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  canny = cv2.Canny(blur.copy(), threshold1, threshold2)
  return canny

#HSV values
max_value_H = int(360/2)
max_value = 255
low_H = 0; high_H = max_value_H
low_S = 0; high_S = max_value
low_V = 0; high_V = max_value

cv2.namedWindow('Canny_settings')
cv2.createTrackbar('threshold1','Canny_settings',1,249,nothing)
cv2.createTrackbar('threshold2','Canny_settings',1,249,nothing)

# cv2.namedWindow('Stick_HSV_Color_settings')
# cv2.createTrackbar('LowH','Stick_HSV_Color_settings',  low_H, max_value_H,nothing)
# cv2.createTrackbar('HighH','Stick_HSV_Color_settings', high_H, max_value_H,nothing)
# cv2.createTrackbar('LowS','Stick_HSV_Color_settings',  low_S,max_value,nothing)
# cv2.createTrackbar('HighS','Stick_HSV_Color_settings', high_S,max_value,nothing)
# cv2.createTrackbar('LowV','Stick_HSV_Color_settings',  low_V,max_value,nothing)
# cv2.createTrackbar('HighV','Stick_HSV_Color_settings', high_V,max_value,nothing)  

def stickColorIsolate(image):
  # lowh = cv2.getTrackbarPos('LowH','Stick_HSV_Color_settings')
  # highh = cv2.getTrackbarPos('HighH','Stick_HSV_Color_settings')
  # lows = cv2.getTrackbarPos('LowS','Stick_HSV_Color_settings')
  # highs = cv2.getTrackbarPos('HighS','Stick_HSV_Color_settings')
  # lowv = cv2.getTrackbarPos('LowV','Stick_HSV_Color_settings')
  # highv = cv2.getTrackbarPos('HighV','Stick_HSV_Color_settings')
  frame = cv2.GaussianBlur(image, (5, 5), 0)
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  cv2.imshow("HSV", hsv)
  low_skin = np.array([0, 0, 173])
  up_skin = np.array([180, 255, 255])
  mask = cv2.inRange(image, low_skin, up_skin)
  edges = cv2.Canny(mask.copy(), 33, 0)
  cv2.imshow("mask", mask)
  cv2.imshow("mask>edges", edges)
  
  return edges
 
image = cv2.imread('img.jpg')

while (1):

  threshold1 = cv2.getTrackbarPos('threshold1','Canny_settings')
  print("threshold1: " + str(threshold1))
  threshold2 = cv2.getTrackbarPos('threshold2','Canny_settings')
  print("threshold2: " + str(threshold2))
  board = image.copy()
  canny_image = canny(board)#, threshold1= threshold1, threshold2 = threshold2)
  stick_image = stickColorIsolate(board)
  cv2.imshow("Canny_Image", canny_image)
  #cv2.imshow("Stick_Image", stick_image)
  if cv2.waitKey(100) == 13:
    break

cv2.destroyAllWindows()
