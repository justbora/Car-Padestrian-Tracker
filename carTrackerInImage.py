import cv2

# our pre-trained car classifier
classifier_file='car_detector.xml'

# create opencv image
img=cv2.imread('highway-5c6e5194c9e77c0001cda269.jpg')
grayscalled_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker=cv2.CascadeClassifier(classifier_file)

# detect cars
cars_coordinates=car_tracker.detectMultiScale(grayscalled_img)
print(cars_coordinates)

for (x,y,w,h) in cars_coordinates:
    cv2.rectangle(img,(x,y),(x+h,y+h),(0,255,0),2)

# display the image with the faces spotted 
cv2.imshow('Gaurav Bora Car Detector', img)

# dont autoclose (wait  here in the code and listen for a key press)
cv2.waitKey()