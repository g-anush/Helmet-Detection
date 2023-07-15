#importing Python Libraries

import cv2                 # for image pre-processing
import torch               # for loading and using model
import numpy as np         # Pyhton library for mathematical operations 


# Model path
path='D:/Download/yolov5safetyhelmet-main/yolov5safetyhelmet-main/best.pt'

# Loading dataset into model
model = torch.hub.load('ultralytics/yolov5', 'custom',path, force_reload=True)

# Input video
cap=cv2.VideoCapture('helmet.mp4')
count=0
while True:
    
    # function that read video in frame format
    ret,frame=cap.read()
    
    # if video ends then break
    if not ret:
        break
    
    # Checking input video for every 3rd frame 
    count += 1
    if count % 3 != 0:
        continue
    
    # Output video frame size and 
    frame=cv2.resize(frame,(1020,600))
    # Using model in given frame
    results=model(frame)
    # Capturing the desired result obtained into result frame
    frame=np.squeeze(results.render())
    
    #results=model(frame)
    # Showing the output window
    cv2.imshow("FRAME",frame)
    
    # Program ending command 
    if cv2.waitKey(1)&0xFF==27:
        break
    
# Erasing the used memory    
cap.release()
cv2.destroyAllWindows()