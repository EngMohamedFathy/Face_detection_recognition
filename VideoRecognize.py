import numpy as np;
import os;
import math;

from matplotlib import pyplot as plt;



import cv2;
print cv2.__version__;

class VideoCamera(object):
    def __init__(self,index=0):
        self.video=cv2.VideoCapture(index)
        self.index=index;
        print self.video.isOpened();

    def __del__(self):
        self.video.release();

    def getFrame(self,in_grayscale=False):
        _, frame = self.video.read();
        if in_grayscale:
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
        return frame;

#----------------------------------------
#second useful class--------------------
class faceDetector(object):
    def __init__(self,xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path);

    def detect(self,image,biggest_only=True):
        scale_factor = 1.2;
        min_neighbors = 5;
        min_size = (30, 30);
        bigest_only = True;

        face_coord = self.classifier.detectMultiScale(image, 1.2, int(min_neighbors));

        return face_coord;
#-----------------------------------------
#my class maked by me----------------------
class drawRectOverFace_andShow(object):
    def draw(self,image,face_coord):
        for (x, y, w, h) in face_coord:
            cv2.rectangle(image, (x, y), (x + w, y + h), (150, 150, 0), 8);
#-----------------------------------------

#method to cut face only from image----------
def cut_faces(image,faces_coord):
    faces=[];

    for (x, y, w, h) in faces_coord:
        w_rm=int(0.2*w/2)
        faces.append(image[y:y+h ,x + w_rm:x+w-w_rm])

    return faces;
#-------------------------------------------------

#method for normlize instnesity----------
def normalize_intensity(images):
    images_norm=[];
    for image in images:
        is_color=len(image.shape)==3;
        if is_color:
            image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);
        images_norm.append(cv2.equalizeHist(image));
    return images_norm;
#----------------------------------------


#method for resize --------------------------
def resize(images ,size=(50,50)):
    images_norm=[];
    for image in images:
        if image.shape < size :
            image_norm=cv2.resize(image,size,interpolation=cv2.INTER_AREA);
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)


        images_norm.append(image_norm);
    return images_norm;
#---------------------------------------


#777777777777777777777777777777777
#method for bring all together for normalize face and draw rectangle-----------
def normalize_faces(frame,faces_coord):

    faces = cut_faces(frame, faces_coord);
    faces = normalize_intensity(faces);
    faces = resize(faces);
    return faces;

def draw_rectangle(image,coords):

    for (x, y, w, h) in coords:
        cv2.rectangle(image, (x, y), (x + w, y + h), (150, 150, 0), 8);
#-------------------------------------------------------------------------------


def collect_dataset():
    images=[];
    labels=[]
    labels_dic={}

    people=[person for person in os.listdir("people\\")]
    for i,person in enumerate(people):
        labels_dic[i]=person
        for image in os.listdir("people\\"+ person):
            images.append(cv2.imread("people\\"+ person+'\\'+image,0))
            labels.append(i);
    return (images ,np.array(labels),labels_dic)
#----------------------------------------------------------
images,labels,labels_dic=collect_dataset();

rec_eig=cv2.face.EigenFaceRecognizer_create();
rec_eig.train(images,labels)

'''
#need at least two people
rec_fisher=cv2.face.FisherFaceRecognizer_create();
rec_fisher.train(images,labels);
'''
rec_lbph =cv2.face.LBPHFaceRecognizer_create();
rec_lbph.train(images,labels);

print "models trained succesfully"

detector = faceDetector("haarcascades/haarcascade_frontalface_default.xml");
webcam = VideoCamera(1);
'''
cv2.namedWindow("the image", cv2.WINDOW_NORMAL);

while True:
    frame = webcam.getFrame();
    faces_coord = detector.detect(frame);
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord);
        for i,face in enumerate(faces):#for detect face
            pred, conf = rec_lbph.predict(face)
            print ("eigen faces ->perdiction " + labels_dic[pred].capitalize() + "  confidence :" + str(round(conf)));
            cv2.putText(frame,labels_dic[pred],(faces_coord[i][0],faces_coord[i][1]-10),cv2.FONT_HERSHEY_PLAIN,3,(66,53,243),2)
        draw_rectangle(frame,faces_coord)
    cv2.putText(frame,"ESC TO EXIT",(5,frame.shape[0]-5),cv2.FONT_HERSHEY_PLAIN,3,(66,53,243),2)
    cv2.imshow("the image",frame)#live video
    if (cv2.waitKey(20) & 0xff == ord('q')):
        cv2.destroyAllWindows()
        break;
'''
#22222222222222222222222222222222222222222222222222222 in thare are unknown person

while True:
    frame = webcam.getFrame();
    faces_coord = detector.detect(frame);
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord);
        for i,face in enumerate(faces):#for detect face
            pred, conf = rec_lbph.predict(face)
            print ("eigen faces ->perdiction " + labels_dic[pred].capitalize() + "  confidence :" + str(round(conf)));




            threshold=150;
            if conf<threshold: #apply threshold
                cv2.putText(frame, labels_dic[pred], (faces_coord[i][0], faces_coord[i][1] - 10),cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)

            else:
                cv2.putText(frame, "unKnown", (faces_coord[i][0], faces_coord[i][1] - 10),cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)




        draw_rectangle(frame,faces_coord)
    cv2.putText(frame,"ESC TO EXIT",(5,frame.shape[0]-5),cv2.FONT_HERSHEY_PLAIN,3,(66,53,243),2)
    cv2.imshow("the image",frame)#live video
    if (cv2.waitKey(20) & 0xff == ord('q')):
        cv2.destroyAllWindows()
        break;














