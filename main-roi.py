import cv2  # import libary


# We need to do object tracking to know where the bike or Vehicles are before getting to roi
from tracker import *


# There are many Distance  like Euclidean, Logarithm, Corrolation coefficient ,etc
# but here we are going to use EuclideanDistanceTracker
#create tracker object
tracker = EuclideanDistTracker()

# Display video with opencv
cap = cv2.VideoCapture("highway.mp4")



# Object detection from stable Camera

# Since we are not getting accurate detection we can include the history and value Threshold
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()

    # we need know the height and width
    height, width, _ =  frame.shape
   # print(height,width)  # 720, 1280

    #Extract Region of Interest (H, W)
    roi = frame[340 : 720, 500 : 800]


   #Object Detection from the Mask
    #mask = object_detector.apply(frame)

    #  1. 0 Object Detection

    
    #Detemining the ROI
    mask = object_detector.apply(roi)

    # Remove all the shadow , keep all the white and remove gray  <254(which is black and
    # live only white 255
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    

    # We use open cv and extract information from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # we need to put the inside array
    detections = []
    for cnt in contours:
        

        #calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:

            # Before we shown in frame now in ROI
           # cv2.drawContours(frame, [cnt], -1, (0, 255,0),2)

           
            # Since we got Contour here next we need bbox
           # cv2.drawContours(roi, [cnt], -1, (0, 255,0),2)

           # we need draw bounding box arounf the objects
            x, y, w, h = cv2.boundingRect(cnt)

            # we move it to tracking 
            #cv2.rectangle(roi, (x, y), (x + w, y + h), (0,255, 0), 3)

            
           # print(x, y, w, y)
            detections.append([x ,y, w,h])

     # 2 . 0 Object Tracking
    boxes_ids = tracker.update(detections)
    #print(boxes_ids)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id

        # we have to detect rectangle now from tracking not from Detection.
        cv2.putText(roi, str(id),(x ,y - 15), cv2.FONT_HERSHEY_PLAIN, 1,(255, 0, 0), 1)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0,255, 0), 3)
        
    

     #detections
   # print(detections)
    #show ROI
   # cv2.imshow("ROI", roi)
      
    cv2.imshow("Frame", frame)
    
    #show the mask
   # cv2.imshow("Mask",mask)

   # key = cv2.waitKey(0)
    key = cv2.waitKey(30)
    if key ==27:
        break

cap.release()
cv2.destroyAllWindow()

    

    
