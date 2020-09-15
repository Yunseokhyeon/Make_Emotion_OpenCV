from util import reg_face2, image_bright, image_dark
import cv2
import os
import glob

skin = "white"
faces_folder_path = "./" + skin
count = 0
# 사진속에서 얼굴들을 잘라서 밝기를 높이거나 줄인다.
for f in glob.glob(os.path.join(faces_folder_path, "*")):
    img = cv2.imread(f)

    faces = reg_face2(img)

    for face in faces:
        #print("./after"+skin+"/" + skin + "_"+ str(count) + ".png")
        face = image_bright(face)
        cv2.imwrite("./after"+skin+"/" + skin + "_"+ str(count) + ".png", face)
        count = count +1



