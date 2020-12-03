import numpy as np;
import os;
import math;

from matplotlib import pyplot as plt;


import cv2;
print cv2.__version__;

#22222222222222222222222222222
webcam=cv2.VideoCapture(0);
ret,frame=webcam.read();
print ret;
webcam.release();

#33333333333333333333333333333
cv2.startWindowThread();

cv2.namedWindow("cvtutorial window",cv2.WINDOW_NORMAL);
cv2.imshow("the image",frame);
cv2.waitKey();
cv2.destroyAllWindows();
#444444444444444444444444444444
#555555555555555555555555555555
plt.imshow(frame);
plt.show();

#6666666666666666666666666666666
frame_RGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);
plt.imshow(frame_RGB);
plt.axis("off");
plt.show();

#777777777777777777777777777777777
picture_rgb=cv2.imwrite('1.jpg',1);
picture_gray=cv2.imwrite('1.jpg',1);
picture2=cv2.imread("2.jpg")

picture=np.hstack((picture_rgb,picture_gray));
plt.axis("off");
plt.title("RGB  grayscale")  ;
plt.imshow(picture,cmap="Greys_r")
plt.show();




