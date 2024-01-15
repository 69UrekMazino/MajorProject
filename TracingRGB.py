import cv2
import numpy as np
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0) 
# Variables for RGB traces extraction
red_values = []
green_values = []
blue_values = []
while True:
    ret, frame = cap.read()   
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        forehead_top = y + int(0.15 * h)
        forehead_bottom = y + int(0.35 * h)
        forehead_left = x + int(0.2 * w)
        forehead_right = x + int(0.8 * w)
        forehead_ROI = frame[forehead_top:forehead_bottom, forehead_left:forehead_right]
        # Split channels to extract Red, Green, Blue components
        b, g, r = cv2.split(forehead_ROI)
        #this part Store mean intensity values for each channel
        red_values.append(np.mean(r))
        green_values.append(np.mean(g))
        blue_values.append(np.mean(b))
        cv2.rectangle(frame, (forehead_left, forehead_top), (forehead_right, forehead_bottom), (0, 255, 0), 2)
        cv2.imshow('Forehead ROI', forehead_ROI)
    cv2.imshow('Forehead Detection', frame)
    plt.clf()
    plt.plot(red_values, label='Red')
    plt.plot(green_values, label='Green')
    plt.plot(blue_values, label='Blue')
    plt.xlabel('Frames')
    plt.ylabel('Intensity')
    plt.title('Live RGB Traces from Forehead ROI')
    plt.legend()
    plt.pause(0.01)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
