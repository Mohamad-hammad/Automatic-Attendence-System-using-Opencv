import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os


def FindEncodings(images):
    encodeList=[]
    for image in images:
        image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
    return encodeList

def Mark_Attendence(name):
    with open('attendence.csv','r+') as Myfile:
        Temp_List = Myfile.readlines()
        name_List = []
        for line in Temp_List:                          #append all names in the file in name_list
            Line = line.split(',')
            name_List.append(Line[0])
        if name not in name_List:                       #check if name already exists in file or not
            now = datetime.now()                        #get time and date
            dtString = now.strftime('%H:%M:%S')
            Myfile.writelines(f'\n{name},{dtString}')        #write in file

path = 'source images'
images = []
names = []
mylist = os.listdir(path)
for name in mylist:                                                 #read images from source and append names in names list
    current_Image=cv2.imread(f'{path}/{name}')
    images.append(current_Image)
    names.append(os.path.splitext(name)[0])
print('list of names:')
print(names)

EncodeListKnown=FindEncodings(images)                                #encodign the images by calling the function
cap = cv2.VideoCapture(0)                                           #capture video using webcam

while True:
    success, image = cap.read()                                       #read image from webcam
    resized_image = cv2.resize(image,(0,0),None,0.25,0.25)                     #resie image
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)                    #convert the color from BGR to RGB

    faces_Current_Frame = face_recognition.face_locations(resized_image)

    Encode_Current_Frame = face_recognition.face_encodings(resized_image,faces_Current_Frame)

    for encodeFace,Face_loccation in zip(Encode_Current_Frame,faces_Current_Frame):        #match the faces from the source images to each face in the frame from the webcam
        matches=face_recognition.compare_faces(EncodeListKnown,encodeFace)
        facedis=face_recognition.face_distance(EncodeListKnown,encodeFace)
        matchindex=np.argmin(facedis)

        if matches[matchindex]:                                                     #if match found then print the name and draw a square aroud the face to locate it
            name = names[matchindex].upper()
            print(name)
            y1,x2,y2,x1=Face_loccation
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(image,(x1,y1),(x2,y2),(0.255,0),2)
            cv2.rectangle(image,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(image,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            Mark_Attendence(name)                                                   #call function to mark attendence
    cv2.imshow('webcam',image)
    cv2.waitKey(1)                  #wait for 1ms
