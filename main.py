import cv2
import numpy as np
import matplotlib.pyplot as plt


def nothing(x):
    pass


cv2.namedWindow('Canny_settings')
cv2.createTrackbar('threshold1','Canny_settings',1,249,nothing)
cv2.createTrackbar('threshold2','Canny_settings',1,249,nothing)
def canny(image, threshold1 = 33,threshold2 = 0) :
  blur = cv2.GaussianBlur(image, (5,5), 0)
  canny = cv2.Canny(blur.copy(), threshold1, threshold2)
  return canny

#HSV values
# max_value_H = int(360/2)
# max_value = 255
# low_H = 0; high_H = max_value_H
# low_S = 0; high_S = max_value
# low_V = 0; high_V = max_value
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
  #cv2.imshow("HSV", hsv)
  low_skin = np.array([0, 0, 173])
  up_skin = np.array([180, 255, 255])
  mask = cv2.inRange(image, low_skin, up_skin)
  #cv2.imshow("mask", mask)
  return mask


# cv2.namedWindow('HoughLine Settings')
# cv2.createTrackbar('threshold','HoughLine Settings',  low_H, max_value_H,nothing)
# cv2.createTrackbar('minLineLength','HoughLine Settings', high_H, max_value_H,nothing)
# cv2.createTrackbar('maxLineGap','HoughLine Settings',  low_S,max_value,nothing)

def stickLineDetection(canny_image, original_image):
  # threshold = cv2.getTrackbarPos('threshold','HoughLine Settings')
  # minLineLength = cv2.getTrackbarPos('minLineLength','HoughLine Settings')
  # maxLineGap = cv2.getTrackbarPos('maxLineGap','HoughLine Settings')
  houghImage = canny_image.copy()
  lines = cv2.HoughLinesP(canny_image, 1, np.pi/180,125, np.array([]),minLineLength=14, maxLineGap=30)
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      cv2.line(original_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
  cv2.imshow("HoughLines", original_image)


# cv2.namedWindow('morph_settings')
# cv2.createTrackbar('kernel','morph_settings',0,300,nothing)
def morph(image):
  #kernelVal = cv2.getTrackbarPos('kernel','morph_settings')
  #kernel = np.ones((kernelVal,kernelVal), np.uint8)
  #print("kernelVal: " + str(kernelVal))
  #opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
  #morph = cv2.erode(stickColorIsolate_img, kernel, iterations=1)
  #morph = cv2.dilate(morph, kernel, iterations=1)
  #cv2.imshow("morph", morph)
  pass
  
image = cv2.imread('img.jpg')

while (1):

  threshold1 = cv2.getTrackbarPos('threshold1','Canny_settings')
  threshold2 = cv2.getTrackbarPos('threshold2','Canny_settings')
  board = image.copy()
  stickColorIsolate_img = stickColorIsolate(board)
  #morph(stickColorIsolate_img)
  canny_image = canny(stickColorIsolate_img, 250, 250)# threshold1= threshold1, threshold2 = threshold2)
  houghStick = stickLineDetection(canny_image, board)

  cv2.imshow("Stick_Image>Color Iso", stickColorIsolate_img)
  cv2.imshow("Canny_Image", canny_image)
  #cv2.imshow("Color Iso>Hough", houghStick)
  if cv2.waitKey(100) == 13:
    break

cv2.destroyAllWindows()