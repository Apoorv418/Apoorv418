#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
faceCascade = cv2.CascadeClassifier(r"C:\Users\APOORV PANDEY\Downloads\haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(r"C:\Users\APOORV PANDEY\Downloads\haarcascade_eye.xml")
cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=10,
            minSize=(5, 5),
            )
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
               
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()


# In[1]:


import numpy as np
import cv2
import os

# Initialize video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Load Haar cascades
face_detector = cv2.CascadeClassifier(r"C:\Users\APOORV PANDEY\Downloads\haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier(r"C:\Users\APOORV PANDEY\Downloads\haarcascade_eye.xml")

# Input user ID
face_id = input('\nEnter user ID and press <return> ==> ')

print("\n[INFO] Initializing face capture. Look at the camera and wait...")

# Initialize individual face sample count
count = 0

while True:
    ret, img = cam.read()
    if not ret:
        print("[ERROR] Failed to capture image.")
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Eye detection within face ROI
        eyes = eye_detector.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Save the captured face
        count += 1
        cv2.imwrite(r"C:\Users\APOORV PANDEY\Desktop\faces2\User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

    cv2.imshow('Face Capture', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' to exit
    if k == 27:
        break
    elif count >= 10:  # Stop after 10 face samples
        break

# Cleanup
print("\n[INFO] Exiting Program and cleaning up...")
cam.release()
cv2.destroyAllWindows()


# In[2]:



import os
import cv2
import numpy as np
from PIL import Image

# Path for face image database
path = r"C:\Users\APOORV PANDEY\Desktop\faces2"

# Ensure `opencv-contrib-python` is installed for this module
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(r"C:\Users\APOORV PANDEY\Downloads\haarcascade_frontalface_default.xml")

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('jpg', 'png', 'jpeg'))]
    faceSamples = []
    ids = []
    
    for imagePath in imagePaths:
        # Open image, convert to grayscale
        PIL_img = Image.open(imagePath).convert('L')  # Grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        # Extract id from filename: assumes filenames like user.1.jpg
        try:
            id = int(os.path.split(imagePath)[-1].split(".")[1])
        except ValueError:
            print(f"Skipping invalid file: {imagePath}")
            continue

        # Detect faces in the image
        faces = detector.detectMultiScale(img_numpy)
        
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    
    return faceSamples, ids

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)

if faces:
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write(r'C:\Users\APOORV PANDEY\Desktop\model\trainer.yml')
    print("\n [INFO] {0} faces trained. Exiting Program.".format(len(np.unique(ids))))
else:
    print("\n [ERROR] No valid face data found. Check your dataset.")


# In[3]:


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(r'C:\Users\APOORV PANDEY\Desktop\model\trainer.yml')
cascadePath = r"C:\Users\APOORV PANDEY\Downloads\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None'] 
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




