import cv2
import numpy as np
import matplotlib.pyplot as plt


def nothing(x):
    pass

# cv2.namedWindow('Canny_settings')
# cv2.createTrackbar('threshold1','Canny_settings',1,249,nothing)
# cv2.createTrackbar('threshold2','Canny_settings',1,249,nothing)

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
  return lines


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

def make_coordinates(image, fit_lines_average):
  slope, intercept = fit_lines_average

  y1 = image.shape[0]
  y2 = int(y1*(4/5))
  x1 = int((y1 - intercept)/slope)
  x2 = int((y2 - intercept)/slope)
  return np.array([x1,y1,x2,y2])


def average_slope_intercept(image , lines):
  fitLines = [] 

  for line in lines:
    x1, y1, x2, y2 = line.reshape(4)
    parameters = np.polyfit((x1,x2), (y1,y2), 1)
    slope = parameters[0]
    intercept = parameters[1]
    fitLines.append((slope, intercept))
    
  fit_lines_average = np.average(fitLines, axis = 0)
  
  return make_coordinates(image, fit_lines_average)

def display_stick(image, line):
  stickLine_image = np.zeros_like(image)
  if line is not None:
    x1, y1, x2, y2 = line.reshape(4)
    print('x1: %d, y1:%d, x2:%d, y2:%d'%(x1, y1, x2, y2) )
    cv2.line(stickLine_image, (x1,y1), (x2,y2), (255,0,0), 5)     
  return stickLine_image


image = cv2.imread('img.jpg')
plt.imshow(image)
plt.show()

while (1):

  threshold1 = cv2.getTrackbarPos('threshold1','Canny_settings')
  threshold2 = cv2.getTrackbarPos('threshold2','Canny_settings')
  board = image.copy()
  #1
  stickColorIsolate_img = stickColorIsolate(board)
  #cv2.imshow("Stick_Image>Color Iso", stickColorIsolate_img)
  
  #morph(stickColorIsolate_img)
  
  #2
  canny_image = canny(stickColorIsolate_img, 250, 250)# threshold1= threshold1, threshold2 = threshold2)
  #cv2.imshow("Canny_Image", canny_image)
  
  #3
  lines = stickLineDetection(canny_image, board)
  line = average_slope_intercept(board, lines)
  stick_average = display_stick(board, line)
  averaged_combined_image = cv2.addWeighted(board, 0.8, stick_average, 1, 1)
  
  cv2.imshow("Averaged Hough", averaged_combined_image)

  #cv2.imshow("Color Iso>Hough", houghStick)
  if cv2.waitKey(100) == 13:
    break

cv2.destroyAllWindows() 



