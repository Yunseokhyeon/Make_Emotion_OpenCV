#for trans_image_model
import numpy as np
from skimage import transform, io
import dlib
import cv2

# 모델들이 사용할 수 있게 읽어온 이미지를 원하는 array로 변환
def trans_image_model(image):
    # cv2는 이미지를 읽어올때 GBR로 읽어온다. 이를 RGB로 변환
    image_for_model = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_for_model = np.array(image_for_model).astype('float32')/255
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.png",gray)
    # 안경, 선글라스, 마스크 모델용 이미지
    # RGB 3채널 + 학습사진크기 : 128 * 128
    RGBimage = transform.resize(image_for_model, (128, 128, 3))

    RGBimage = np.expand_dims(RGBimage, axis=0)

    # 감정 모델용 이미지 
    # gray 스케일로 인한 1채널 + 학습사진크기 : 48 * 48 
    GRAYimage = transform.resize(image_for_model, (48, 48, 1))
    
    GRAYimage = np.expand_dims(GRAYimage, axis=0)
    
    

    return (RGBimage, GRAYimage)

# 이미지에서 얼굴인식하고 추출
def reg_face(image):
    
    prototxtPath = "./face_detector/deploy.prototxt"
    weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    (h, w) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    
    
    # 1x1xNx7 차원 행렬
    # N : N개의 얼굴 후보군
    # 7 : 7 개의 데이터
    #     [2] : 신뢰도 / [3~6] : 사각형의 얼굴영역
    detections = faceNet.forward()

    locs = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # 신뢰도 80% 이상일때
        if confidence > 0.8:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
	    	
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            locs.append((startX, startY, endX, endY))
    
    if len(locs) > 1:
        print("사람이" , len(locs), "입니다")
        return image, -1

    return image[locs[0][1]:locs[0][3], locs[0][0]:locs[0][2]], 0

# 얼굴사진에 얼굴부위 붙이기
# 얼굴사진, 얼굴부위, 감정, 붙일 좌표
def add_face_part(face, emotion, face_part, X, Y):
    part = cv2.imread('./emoticon/'+emotion + '/' + face_part + '.png', -1)

    h, w = part.shape[:2]
    b,g,r,a = cv2.split(part) # RGB 채널 + alpha 채널
    mask = np.dstack((a,a,a))
    part = np.dstack((b,g,r))

    canvas = face[Y:Y+h, X:X+w]
    imask = mask>0
    canvas[imask] = part[imask]

# 이미지에서 얼굴들 추출
def reg_face2(image):
    
    prototxtPath = "./face_detector/deploy.prototxt"
    weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    (h, w) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (224, 224),
		(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    
    faces = []
    
    # 1x1xNx7 차원 행렬
    # N : N개의 얼굴 후보군
    # 7 : 7 개의 데이터
    #     [2] : 신뢰도 / [3~6] : 사각형의 얼굴영역
    detections = faceNet.forward()

    locs = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # 신뢰도 80% 이상일때
        if confidence > 0.8:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
	    	
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.resize(face, (128, 128))
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    return faces

# 이미지 밝게
def image_bright(image):
    M = np.ones(image.shape, dtype = "uint8") * 65
    added = cv2.add(image, M)
    return added

# 아미지 어둡게
def image_dark(image):
    M = np.ones(image.shape, dtype = "uint8") * 120
    added = cv2.subtract(image, M)
    return added