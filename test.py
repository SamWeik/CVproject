import cv2
import numpy as np

capture = cv2.VideoCapture(0)
ret1,frame1= capture.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (25, 25), 0)
 
while(True):
     
    ret, frame = capture.read()

    #Flips Video
    mirror = cv2.flip(frame, 1)

    #Edge Detection
    edges = cv2.Canny(mirror, 150, 100)

    #Color Detection
    hsvFrame = cv2.cvtColor(mirror, cv2.COLOR_BGR2HSV)
    red_lower = np.array([0,100,100])
    red_upper = np.array([5,255,255])
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    green_lower = np.array([25, 52, 72]) 
    green_upper = np.array([102, 255, 255]) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    blue_lower = np.array([94, 80, 2]) 
    blue_upper = np.array([120, 255, 255]) 
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 

    redResult = cv2.bitwise_and(mirror,mirror,mask=red_mask)
    blueResult = cv2.bitwise_and(mirror,mirror,mask=blue_mask)
    greenResult = cv2.bitwise_and(mirror,mirror,mask=green_mask)

    combine = cv2.bitwise_or(blueResult,greenResult)
    combine = cv2.bitwise_or(redResult, combine)

    #Motion Detection
    gray2 = cv2.cvtColor(mirror, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    deltaframe=cv2.absdiff(gray1,gray2)
    threshold = cv2.threshold(deltaframe, 25, 255, cv2.THRESH_BINARY)[1]
    threshold = cv2.dilate(threshold,None)
    countour,heirarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in countour:
        if cv2.contourArea(i) < 50:
            continue
        (x, y, w, h) = cv2.boundingRect(i)
        cv2.rectangle(mirror, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #cv2.imshow('Normal', mirror)
    cv2.imshow('Edges Detector', edges)
    cv2.imshow('Color Detector', combine)
    cv2.imshow('Motion Detector', mirror)
     
    if cv2.waitKey(1) == 27:
        break
 
capture.release()
cv2.destroyAllWindows()