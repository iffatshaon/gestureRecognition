import cv2
import numpy as np
from PIL import Image, ImageOps
from keras.preprocessing import image
from keras.models import load_model
i=0
key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
saved_path = "VGG_cross_validated.h5"
model = load_model(saved_path)
oval = 122
font = cv2.FONT_HERSHEY_SIMPLEX
while True:

    check, frame = webcam.read()
    frame = cv2.flip(frame,1)
    framecp = frame.copy()
    kernel = np.ones((3,3),np.uint8)
    cv2.rectangle(framecp,(390,80),(640,380),(70,0,70), 3)
    cv2.circle(framecp, (515,230), 3, (70,0,70), 3)
    subimg = frame[80:380,390:640]
    hsv = cv2.cvtColor(subimg, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask,kernel,iterations = 4)
    mask = cv2.GaussianBlur(mask,(5,5),100)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)[1]
    mask = cv2.erode(mask,kernel,iterations = 3)
    mask = cv2.dilate(mask,kernel,iterations = 3)
    imgs = Image.fromarray(mask)
    imgs = imgs.convert('RGB')
    imgs = ImageOps.fit(imgs, (224,224), Image.ANTIALIAS)
    imgs = np.expand_dims(imgs,axis=0)
    val = model.predict(imgs)
    res = np.where(val == np.amax(val))
    
    if res[1] == [0]:
        cv2.putText(framecp,"Fist",(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif res[1] == [1]:
        cv2.putText(framecp,"L",(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif res[1] == [2]:
        cv2.putText(framecp,"OK",(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif res[1] == [3]:
        cv2.putText(framecp,"Palm",(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif res[1] == [4]:
        cv2.putText(framecp,"Peace",(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    oval = res[1]
    cv2.imshow("Testing",mask)
    cv2.imshow("Capturing", framecp)
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break