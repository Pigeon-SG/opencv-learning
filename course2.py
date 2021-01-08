import cv2
import numpy as np
# img 里的shape 都是先height 然后 width
img = np.zeros((512,512,3),np.uint8)
cv2.rectangle(img,(200,100),(0,0),(100,100,255),-1)
cv2.putText(img,"hello opencv",(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(100,100,100),1)


cv2.namedWindow("1",cv2.WINDOW_NORMAL)
cv2.resizeWindow("1",1000,1000)
cv2.imshow("1",img)
cv2.waitKey(0)