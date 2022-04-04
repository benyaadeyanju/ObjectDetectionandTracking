import cv2  # import libary

# Display video with opencv
cap = cv2.VideoCapture("highway.mp4")



# Object detection from stable Camera
object_detector = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    # we need know the height and width
    height, width, _ =  frame.shape
   # print(height,width)  # 720, 1280

    #Extract Region of Interest (H, W)
    roi = frame[340 : 720, 500 : 800]


   #Object Detection from the Mask
    mask = object_detector.apply(frame)

    # We use open cv and extract information from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        

        #calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(frame, [cnt], -1, (0, 255,0),2)
    


    #show ROI
    cv2.imshow("ROI", roi)

     
    cv2.imshow("Frame", frame)
    
    #show the mask
    cv2.imshow("Mask",mask)

    key = cv2.waitKey(30)
    if key ==27:
        break

cap.release()
cv2.destroyAllWindow()
