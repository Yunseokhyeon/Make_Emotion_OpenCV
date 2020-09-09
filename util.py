#for trans_image_model
import numpy as np
from skimage import transform

# for face_align
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
import dlib
import cv2

# 모델들이 사용할 수 있게 읽어온 이미지를 원하는 array로 변환
def trans_image_model(image):
    # cv2는 이미지를 읽어올때 GBR로 읽어온다. 이를 RGB로 변환
    image_for_model = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_for_model = np.array(image_for_model).astype('float32')/255

    # 안경, 선글라스, 마스크 모델용 이미지
    # RGB 3채널 + 학습사진크기 : 128 * 128
    RGBimage = transform.resize(image_for_model, (128, 128, 3))
    RGBimage = np.expand_dims(RGBimage, axis=0)

    # 감정 모델용 이미지 
    # gray 스케일로 인한 1채널 + 학습사진크기 : 48 * 48 
    GRAYimage = transform.resize(image_for_model, (48, 48, 1))
    GRAYimage = np.expand_dims(GRAYimage, axis=0)
   
    return (RGBimage, GRAYimage)


def face_align(image):
    predictor_model = "./landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_model)
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)

    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x+w], width=256)
        faceAligned = fa.align(image, gray, rect)
        return faceAligned