import cv2
import matplotlib.pyplot as plt
def plt_show(img):
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
img = cv2.imread(r"pics/Mrjintaixian.jpg")
img_resize = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
gray = cv2.cvtColor(img_resize,cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(r"classifier\haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5)
face_num = 0



if len(faces) > 0:
    for faceRect in faces:
        face_num += 1
        x,y,w,h = faceRect
        cv2.rectangle(img_resize, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # 截取图像
        cropimg = gray[y:y + h, x:x + w]
        cv2.imwrite("pics_processed/xx_face" + str(face_num) + ".jpg", cropimg)