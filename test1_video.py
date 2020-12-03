import numpy as np;
import os;
import math;

from matplotlib import pyplot as plt;


import cv2;
print cv2.__version__;

#11111111111111111
'''
webcam =cv2.VideoCapture(0);
print webcam.isOpened();

#2222222222222222222222222222222
cv2.namedWindow("the image",cv2.WINDOW_NORMAL);

while True:
    _,frame=webcam.read();
    cv2.imshow("the image",frame);

    if(cv2.waitKey(20) & 0xff==27):
        break

cv2.destroyAllWindows();

'''
#3333333333333333333333333333333
'''videoCam =cv2.VideoCapture("mysql instlling.avi");
print videoCam.isOpened();

cv2.namedWindow("the image",cv2.WINDOW_NORMAL);
cv2.resizeWindow("the image",850,480)

while videoCam.isOpened():
    _,frame=videoCam.read();
    cv2.imshow("the image",frame);

    if(cv2.waitKey(20) & 0xff==ord('q')):
        break
videoCam.release();
cv2.destroyAllWindows();
'''
#44444444444444444444444444444444444444
'''
videoCam =cv2.VideoCapture("mysql instlling.avi");
print videoCam.isOpened();

cv2.namedWindow("the image",cv2.WINDOW_NORMAL);
cv2.resizeWindow("the image",850,480)

while videoCam.isOpened():
    ret,frame=videoCam.read();

    if not ret:
        break;f

    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
    frame=cv2.flip(frame,1)



    cv2.imshow("the image",frame);

    if(cv2.waitKey(20) & 0xff==ord('q')):
        break
videoCam.release();
cv2.destroyAllWindows();
'''
#55555555555555555555555555555555555555
'''
videoCam =cv2.VideoCapture(0);
print videoCam.isOpened();

cv2.namedWindow("the image",cv2.WINDOW_NORMAL);
cv2.resizeWindow("the image",850,480)

fourcc=cv2.VideoWriter_fourcc(*'XVID');
video=cv2.VideoWriter('video12.avi',fourcc,7,(640,480));


while videoCam.isOpened():
    ret,frame=videoCam.read();
    video.write(frame);

    if not ret:
        break;

    cv2.imshow("the image",frame);

    if(cv2.waitKey(7) & 0xff==ord('q')):
        break
videoCam.release();
video.release();
cv2.destroyAllWindows();
'''


