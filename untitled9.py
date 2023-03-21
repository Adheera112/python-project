from mouse import *
import cv2

cap = cv2.VideoCapture(0)
hd=Tracking()

while True:
    # Get image frame
    success, img = cap.read()
    hd.run()
    cv2.imshow("Image",img)
   
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()