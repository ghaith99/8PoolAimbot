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

# max_value_H = int(360/2)
# max_value = 255
# cv2.namedWindow('HoughLine Settings')
# cv2.createTrackbar('threshold','HoughLine Settings',  0, max_value_H,nothing)
# cv2.createTrackbar('minLineLength','HoughLine Settings', 0, max_value_H,nothing)
# cv2.createTrackbar('maxLineGap','HoughLine Settings',  0,max_value,nothing)

def stickLineDetection(canny_image, original_image):
  # threshold = cv2.getTrackbarPos('threshold','HoughLine Settings')
  # minLineLength = cv2.getTrackbarPos('minLineLength','HoughLine Settings')
  # maxLineGap = cv2.getTrackbarPos('maxLineGap','HoughLine Settings')
  houghImage = np.zeros_like(original_image)
#  lines = cv2.HoughLinesP(canny_image, 1, np.pi/180,threshold, np.array([]),minLineLength=minLineLength, maxLineGap=maxLineGap)
#  lines = cv2.HoughLinesP(canny_image, 1, np.pi/180,125, np.array([]),minLineLength=14, maxLineGap=30)
  lines = cv2.HoughLinesP(canny_image, 1, np.pi/180,2, np.array([]),minLineLength=74, maxLineGap=19)
  size = [] 
  longLine = None
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line[0]
      size.append((y2-y1)**2+ (x2-x1)**2)  #calculate line length
  
    #Filter out only longest line coordinates
    longLine = lines[size.index(max(size))].reshape(4)
    x1, y1, x2, y2 = longLine
    #print("Slope: "+ str(np.polyfit((x1,x2), (y1,y2), 1)[0]))
    cv2.line(houghImage, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.line(original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

  cv2.imshow("HoughLines", houghImage)
  cv2.imshow("Original+HoughLines", original_image)
  return longLine


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

  y1 = int(intercept)
  y2 = int(y1*(2/5))
  x1 = 40
  x2 = 1000
  #y1 = slope*x1 + intercept
  #y2 = slope*x2 + intercept  
  # x1 = int((y1 - intercept)/slope)
  # x2 = int((y2 - intercept)/slope)
  return np.array([x1,y1,x2,y2])


def longerLine(image, lines) :
  size = []
  for line in lines:
    x1, y1, x2, y2 = line.reshape(4)
    size.append((y2-y1)**2+ (x2-x1)**2)  
  # return only longest line coordinates
  x1, y1, x2, y2 = lines[size.index(max(size))].reshape(4)
  parameters = np.polyfit((x1,x2), (y1,y2), 1)
  slope = parameters[0]
  intercept = parameters[1]

  return make_coordinates(image, [slope, intercept]) 
    
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
    cv2.line(stickLine_image, (x1,y1), (x2,y2), (0,255,0), 5)     
  return stickLine_image

def extendHough(image, line):
  print(line)
  x1, y1, x2, y2 = line
  #print("Slope: "+ str(np.polyfit((x1,x2), (y1,y2), 1)[0]))
  params = np.polyfit((x1,x2), (y1,y2), 1)
  slope = params[0]
  intercept = params[1]
  slopeDir = np.sign(np.polyfit((x1,x2), (y1,y2), 1)[0])
  x2 = 1500   
  x1 = 0
  y1 = int(x1*slope + intercept)
  y2 = int(x2*slope + intercept)
  
   
  extended = image.copy()
  print(str(x1)+"->x1, "+str(y1)+"->y1, "+str(x2)+"->x2, "+str(y2)+"->y2")
  cv2.line(extended, (x1,y1), (x2,y2), (0,255,0), 1)   
  cv2.imshow("Extended", extended)
  

from PIL import ImageGrab

image = cv2.imread('img5.jpg')

# plt.imshow(image)
# plt.show()

while (1):
  printscreen =  ImageGrab.grab()
  image =   np.array(printscreen .getdata(),dtype='uint8')\
      .reshape((printscreen.size[1],printscreen.size[0],3)) 
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
  threshold1 = cv2.getTrackbarPos('threshold1','Canny_settings')
  threshold2 = cv2.getTrackbarPos('threshold2','Canny_settings')
  board = image.copy()
  #1
  stickColorIsolate_img = stickColorIsolate(board)
  cv2.imshow("1- Stick_Image>Color Iso", stickColorIsolate_img)
  
  #morph(stickColorIsolate_img)
  
  #2
  canny_image = canny(stickColorIsolate_img, 250, 250)# threshold1= threshold1, threshold2 = threshold2)
  cv2.imshow("2- Canny_Image", canny_image)
  
  #3
  line = stickLineDetection(canny_image, board)
  if line is not None:
    extendHough(image, line)
   
  # long_combined_image = cv2.addWeighted(board, 0.8, stick_long, 1, 1)
  # cv2.imshow("long Hough", long_combined_image)

  #cv2.imshow("Color Iso>Hough", houghStick)
  if cv2.waitKey(100) == 13:
    break

cv2.destroyAllWindows() 



