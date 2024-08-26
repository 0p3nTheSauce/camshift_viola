import cv2
import numpy as np
import os

def viola_jones(video_capture):
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
    if len(faces) > 0:
      (x, y, w, h) = faces[0]
      cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
      faceROI = frame[y:y+h, x:x+w]
      eyes = eyeCascade.detectMultiScale(faceROI)
      if len(eyes) >= 2:
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
        nose_centre = ((eye_center1[0] + eye_center2[0])//2, (eye_center1[1] + eye_center2[1])//2)
        xn = nose_centre[0] - 10
        yn = nose_centre[1] - 10
        wn = 20
        hn = 20
        cv2.rectangle(frame, (xn, yn), (xn+wn, yn+hn), (0, 0, 255), 2)
        #Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
      #When the 's' key is pressed, return the nose centre
      return nose_centre

def camshift(video_capture, nose_centre):
  ret, frame = video_capture.read()
  #Camshift
  #We want our roi to be 10 x 10 pixels
  x = nose_centre[0] - 10 #Top left corner
  y = nose_centre[1] - 10
  w = 20
  h = 20
  #Initial location of window
  track_window = (x, y, w, h)
  #Set up the ROI for tracking
  roi = frame[y:y+h, x:x+w]
  hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
  #binary mask pixels that fall within the range
  mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
  #create histogram
  roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
  cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
  peak_hue = np.argmax(roi_hist)
  #Setup the termination criteria, either 10 iteration or move by at least 1 pt
  term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
  while True:
    ret, frame = video_capture.read()
    if not ret:
      break
    #Probability pixel belongs to the object
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    #Apply camshift to get the new location
    rect, track_window = cv2.CamShift(dst, track_window, term_crit) 
    #Draw it on image
    pts = cv2.boxPoints(rect)
    pts = pts.astype(np.int64)  # or np.int32
    img2 = cv2.polylines(frame,[pts],True, 255,2)
    cv2.imshow('img2',img2)
    choice = cv2.waitKey(1) & 0xFF
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break
    if choice == ord('s'):
      return peak_hue
    
      
def skinsegmentation(video_capture, peak_hue):
  # Define a range around the peak hue
  delta_hue = 40 
  min_saturation = 50
  min_value = 50
  max_saturation = 255
  max_value = 255 
  lower_skin = np.array([peak_hue - delta_hue, min_saturation, min_value]) 
  upper_skin = np.array([peak_hue + delta_hue, max_saturation, max_value])  
  while True:
    ret, frame = video_capture.read()
    if not ret:
      break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # Define the kernel size for the morphological operations
    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel of ones

    # Opening operation
    opened_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
    cv2.imshow('Original', frame)
    cv2.imshow('Skin', skin)
    # Display or use the refined mask
    cv2.imshow('Refined Skin Mask opened', opened_mask)
    choice = cv2.waitKey(1) & 0xFF
    if choice == ord('q'):
      break
    
def main():
  video_capture = cv2.VideoCapture(0)
  nose_centre = viola_jones(video_capture)
  peak_hue = camshift(video_capture, nose_centre)
  skinsegmentation(video_capture, peak_hue)
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
  main()
  