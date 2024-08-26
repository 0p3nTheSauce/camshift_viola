import cv2
import numpy as np
import os

video_capture = cv2.VideoCapture(0)

cascPathface = os.path.dirname(
  cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
  cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"
#create our classifiers
faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)
while True:
  ret, frame = video_capture.read()
  if not ret:
    break
  #Violajones
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60, 60),
    flags=cv2.CASCADE_SCALE_IMAGE
  )
  #Lets start with one face
  (x, y, w, h) = faces[0]
  cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
  faceROI = frame[y:y+h, x:x+w]
  eyes = eyeCascade.detectMultiScale(faceROI)
  #Eye1 
  (xe1, ye1, we1, he1) = eyes[0]
  eye_center1 = (x + xe1 + we1//2, y + ye1 + he1//2)
  radius1 = int(round((we1 + he1)*0.25))
  frame = cv2.circle(frame, eye_center1, radius1, (255, 0, 0), 4)
  #Eye2
  (xe2, ye2, we2, he2) = eyes[1]
  eye_center2 = (x + xe2 + we2//2, y + ye2 + he2//2)
  radius2 = int(round((we2 + he2)*0.25))
  frame = cv2.circle(frame, eye_center2, radius2, (255, 0, 0), 4)
  #Get nose region (centre of eyes)
  nose_centre = (eye_center1[0] + eye_center2[0]//2, eye_center1[1] + eye_center2[1]//2)

  #Camshift
  #We want our roi to be 10 x 10 pixels
  xc = nose_centre[0] - 5 #Top left corner
  yc = nose_centre[1] - 5
  wc = 10
  hc = 10
  #Initial location of window
  track_window = (xc, yc, wc, hc)
  #Set up the ROI for tracking
  cam_roi = frame[y:y+h, x:x+w]
  hsv_roi = cv2.cvtColor(cam_roi, cv2.COLOR_BGR2HSV)
  #binary mask pixels that fall within the range
  mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
  #create histogram
  roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
  cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
  #Setup the termination criteria, either 10 iteration or move by at least 1 pt
  term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
  #probability 
  
  
      

def main():
  print("hello")
  
if __name__ == "__main__":
  main()
  