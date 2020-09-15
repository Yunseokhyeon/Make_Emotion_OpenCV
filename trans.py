from util import reg_face2
import cv2
import os
import glob

skin = "asian"
faces_folder_path = "./" + skin
count = 0
# 폴더의 모든 파일들(사진)을 가지고 얼굴들만 잘라서 일정한크기로 가져와서 저장한다.
for f in glob.glob(os.path.join(faces_folder_path, "*")):
    img = cv2.imread(f)

    faces = reg_face2(img)

    for face in faces:
        #print("./after"+skin+"/" + skin + "_"+ str(count) + ".png")
        cv2.imwrite("./after"+skin+"/" + skin + "_"+ str(count) + ".png", face)
        count = count +1



