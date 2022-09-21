import cv2
from cv2 import VideoCapture
from cv2 import COLOR_BGR2GRAY

video=cv2.VideoCapture('CarDashcamWithPedestrian.mp4')

car_classifierFile='car_detector.xml'
pedestrian_classifierFile='Fullbody_detector.xml'

while True:
    succesfully_read,frame=video.read()
    
    if succesfully_read: 
        grayscalled_vdo=cv2.cvtColor(frame,COLOR_BGR2GRAY)
    else:break
    
    car_detector=cv2.CascadeClassifier(car_classifierFile)
    pedestrian_detector=cv2.CascadeClassifier(pedestrian_classifierFile)
    car_coordinates=car_detector.detectMultiScale(frame)
    pedestrian_coordinates=pedestrian_detector.detectMultiScale(frame)
    
    for (x,y,w,h) in pedestrian_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    for (x,y,w,h) in car_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
            
    cv2.imshow("Gaurav Bora's Car Detector Code!!!",frame)
    
    key=cv2.waitKey(1)
    
    if key==81 or key==113:
        break
    
video.release()