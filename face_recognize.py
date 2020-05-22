"Author:Himanshi"
import cv2, sys, numpy, os

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = '/home/himanshi/PycharmProjects/deep_pulse_project_webcam/datasets'
images=[]
lables=[]

count=0
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        
        path = os.path.join(datasets, subdir)
        for filename in os.listdir(path):
            path1 = path + '/' + filename
            lable = count
            images.append(cv2.imread(path1, 0))
            lables.append(int(lable))
        count += 1
(width, height) = (130, 130)


(images, lables) = [numpy.array(lis) for lis in [images, lables]]

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)


face_cascade = cv2.CascadeClassifier(haar_file)
cam_video = cv2.VideoCapture(0)
while(True):
    (_, im) = cam_video.read()



    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        
        prediction = model.predict(face_resize)
        #cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cropped_image=im[y:y+h,x:x+w]

        

    cv2.imshow('OpenCV', cropped_image)

    key = cv2.waitKey(1)
    # if key == 27:
    #     break
